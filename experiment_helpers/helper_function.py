from collections import Counter

from xplique.concepts.craft import BaseCraft, DisplayImportancesOrder, Factorization, Sensitivity
from sklearn.decomposition import non_negative_factorization
from xplique.attributions.global_sensitivity_analysis import HaltonSequenceRS, JansenEstimator

import numpy as np
import matplotlib.pyplot as plt

def compute_predictions(model, inputs):
    preds_probs = model.predict_proba(inputs)
    preds = np.argmax(preds_probs, axis=1)
    return preds, preds_probs


def l_compute_predictions(model, inputs):
    preds_probs = model.l_predict_proba(inputs)
    preds = np.argmax(preds_probs, axis=1)
    return preds, preds_probs


def estimate_importance(model, craft_instance, drift_basis, inputs, nb_design: int = 32, verbose: bool = False):
    """
    Estimates the importance of each concept for all the classes of interest.

    Parameters
    ----------
    nb_design
        The number of design to use for the importance estimation. Default is 32.
    verbose
        If True, then print the current class CRAFT is estimating importances for,
        otherwise no textual output will be printed.
    """
    y_preds, _ = compute_predictions(model, inputs)

    global_importance = []
    for class_of_interest in np.unique(y_preds):
        filtered_indices = np.where(y_preds == class_of_interest)
        class_inputs = inputs[filtered_indices]

        importances = estimate_importance_helper(craft_instance=craft_instance,
                                                 model=model,
                                                 drift_basis=drift_basis,
                                                 inputs=class_inputs,
                                                 # activations = class_activations,
                                                 class_of_interest=class_of_interest,
                                                 nb_design=nb_design,
                                                 compute_class_importance=True)
        global_importance.append(importances)

    return global_importance


def estimate_importance_helper(craft_instance, model, drift_basis, inputs: np.ndarray = None,
                               class_of_interest: int = None,
                               nb_design: int = 32, compute_class_importance: bool = False) -> np.ndarray:
    """
    Estimates the importance of each concept for a given class, either globally
    on the whole dataset provided in the fit() method (in this case, inputs shall
    be set to None), or locally on a specific input image.

    Parameters
    ----------
    inputs : numpy array or Tensor
        The input data on which to compute the importances.
        If None, then the inputs provided in the fit() method
        will be used (global importance of the whole dataset).
        Default is None.
    nb_design
        The number of design to use for the importance estimation. Default is 32.

    Returns
    -------
    importances
        The Sobol total index (importance score) for each concept.

    """

    coeffs_u = activation_transform(inputs, drift_basis)  # self.transform(inputs)

    masks = HaltonSequenceRS()(len(drift_basis), nb_design=nb_design)
    estimator = JansenEstimator()

    importances = []

    if len(coeffs_u.shape) == 2:
        # apply the original method of the paper

        for coeff in coeffs_u:
            u_perturbated = coeff[None, :] * masks
            a_perturbated = np.reshape(u_perturbated,
                                       (-1, coeff.shape[-1])) @ drift_basis

            _, y_pred = compute_predictions(model, a_perturbated)
            y_pred = y_pred[:, class_of_interest]

            stis = estimator(masks, y_pred, nb_design)

            importances.append(stis)

    importances = np.mean(importances, 0)

    # # Save the results of the computation if working on the whole dataset
    if compute_class_importance:
        most_important_concepts = np.argsort(importances)[::-1]
        # craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts, cmaps=plt.get_cmap('tab20').colors + plt.get_cmap('Set3').colors)
        craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts, cmaps=(plt.get_cmap('tab20').colors + plt.get_cmap('Set3').colors) * 4)
    return importances


def estimate_importance_l(model, craft_instance, drift_basis, inputs, nb_design: int = 32, verbose: bool = False):
    """
    Estimates the importance of each concept for all the classes of interest.

    Parameters
    ----------
    nb_design
        The number of design to use for the importance estimation. Default is 32.
    verbose
        If True, then print the current class CRAFT is estimating importances for,
        otherwise no textual output will be printed.
    """
    y_preds, _ = l_compute_predictions(model, inputs)

    global_importance = []
    for class_of_interest in [0, 1]:
        filtered_indices = np.where(y_preds == class_of_interest)
        class_inputs = inputs[filtered_indices]

        importances = estimate_importance_helper_l(craft_instance=craft_instance,
                                                   model=model,
                                                   drift_basis=drift_basis,
                                                   inputs=class_inputs,
                                                   # activations = class_activations,
                                                   class_of_interest=class_of_interest,
                                                   nb_design=nb_design,
                                                   compute_class_importance=True)
        global_importance.append(importances)

    return global_importance


def estimate_importance_helper_l(craft_instance, model, drift_basis, inputs: np.ndarray = None,
                                 class_of_interest: int = None,
                                 nb_design: int = 32, compute_class_importance: bool = False) -> np.ndarray:
    """
    Estimates the importance of each concept for a given class, either globally
    on the whole dataset provided in the fit() method (in this case, inputs shall
    be set to None), or locally on a specific input image.

    Parameters
    ----------
    inputs : numpy array or Tensor
        The input data on which to compute the importances.
        If None, then the inputs provided in the fit() method
        will be used (global importance of the whole dataset).
        Default is None.
    nb_design
        The number of design to use for the importance estimation. Default is 32.

    Returns
    -------
    importances
        The Sobol total index (importance score) for each concept.

    """

    coeffs_u = activation_transform(inputs, drift_basis)  # self.transform(inputs)

    masks = HaltonSequenceRS()(len(drift_basis), nb_design=nb_design)
    # print(masks.shape)
    estimator = JansenEstimator()

    importances = []

    if len(coeffs_u.shape) == 2:
        # apply the original method of the paper

        for coeff in coeffs_u:
            u_perturbated = coeff[None, :] * masks
            a_perturbated = np.reshape(u_perturbated,
                                       (-1, coeff.shape[-1])) @ drift_basis

            _, y_pred = l_compute_predictions(model, a_perturbated)
            y_pred = y_pred[:, class_of_interest]

            stis = estimator(masks, y_pred, nb_design)

            importances.append(stis)

    importances = np.mean(importances, 0)

    # # Save the results of the computation if working on the whole dataset
    if compute_class_importance:
        most_important_concepts = np.argsort(importances)[::-1]
        # craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts, cmaps=plt.get_cmap('tab20').colors + plt.get_cmap('Set3').colors)
        craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts, cmaps=(plt.get_cmap('tab20').colors + plt.get_cmap('Set3').colors) * 4)
    return importances


def activation_transform(inputs, drift_basis, patches=False, labels=None, activations=None, n_patches=16):
    """
    Transforms the input images into an (N, 320) representation where N is the number of images.

    Parameters:
    - inputs: Input images or data to be transformed.
    - patches: Whether to use patches (if needed for some other functionality).
    - labels: Optional labels for the inputs.
    - activations: Optional pre-computed activations. If None, activations are computed.
    - drift_basis: Predefined basis for NMF.
    - n_patches: Number of patches per image (default is 16).

    Returns:
    - transformed_data: Transformed dataset with shape (N, 320).
    """

    # Step 1: Extract latent activations using drift_craft
    A = inputs  # drift_craft._latent_predict(inputs)  # Assuming A.shape = (N, H, W, D) where D is the activation dimension
    # Step 2: Reshape activations to 2D (flatten the spatial dimensions)
    original_shape = A.shape[:-1]  # Keep original shape to reconstruct later
    re_activations = np.reshape(A, (-1, A.shape[-1]))  # Flatten to (N * H * W, D)
    # Step 3: Apply Non-negative Matrix Factorization (NMF) to reduce dimensionality
    embedding, basis, n_iter = non_negative_factorization(np.array(re_activations),
                                                          n_components=len(drift_basis),
                                                          init='custom',
                                                          update_H=False, solver='mu', H=drift_basis)

    return embedding


from sklearn.metrics import accuracy_score


def local_one_imp_concept_globally_l(craft_instance, importances, labels):
    image_preds = []
    for image_imp in importances:
        max_local_concept = np.argmax(image_imp)

        arguments = []
        for label in [0, 1]:
            argument = np.where(craft_instance.sensitivities[label].most_important_concepts == max_local_concept)[0][0]
            arguments.append(argument)
        image_pred = np.argmin(arguments)
        image_preds.append(image_pred)

    return accuracy_score(image_preds, labels)


def local_imp_concepts_globally_l(craft_instance, importances, num, labels):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    for image_imp in importances:
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))
        image_preds.append(np.argmax(label_imp))

    return accuracy_score(image_preds, labels)


def global_imp_concepts_locally_l(craft_instance, importances, num, labels):
    # This method takes the top 3 global concepts from each class and sums their local importance.
    image_preds = []
    # image_raw =[]
    for image_imp in importances:
        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        image_preds.append(np.argmax(label_imp))

    return accuracy_score(image_preds, labels)


def local_one_imp_concept_globally(craft_instance, importances, labels):
    image_preds = []
    for image_imp in importances:
        max_local_concept = np.argmax(image_imp)

        arguments = []
        for label in [0, 1, 2]:
            argument = np.where(craft_instance.sensitivities[label].most_important_concepts == max_local_concept)[0][0]
            arguments.append(argument)
        image_pred = np.argmin(arguments)
        image_preds.append(image_pred)

    return accuracy_score(image_preds, labels)


def local_imp_concepts_globally(craft_instance, importances, num, labels):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    for image_imp in importances:
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1, 2]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)e
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))
        # print(label_imp)
        image_preds.append(np.argmax(label_imp))

    return accuracy_score(image_preds, labels)


def global_imp_concepts_locally(craft_instance, importances, num, labels):
    # This method takes the top 3 global concepts from each class and sums their local importance. This method gets a 75% accuracy
    image_preds = []
    for image_imp in importances:
        label_imp = []
        for label in [0, 1, 2]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        image_preds.append(np.argmax(label_imp))

    return accuracy_score(image_preds, labels)


def local_imp_concepts_probability(prob_dict, importances, num, labels):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    # image_raw =[]
    for image_imp in importances:
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)
                if top_3 in prob_dict[label].keys():
                    argument = prob_dict[label][top_3]
                    arguments.append(argument)
                else:
                    argument = 0
                    arguments.append(argument)

            label_imp.append(np.sum(arguments))
        image_preds.append(np.argmax(label_imp))
    return accuracy_score(image_preds, labels)


def concept_counter(local_importances, y_preds):
    sample_local_imp = np.array(local_importances)

    # Initialize counters for before drift (0), after drift (1), and both (2)
    before_drift_counter = Counter()
    after_drift_counter = Counter()
    # both_drift_counter = Counter()

    for drift_label in [0, 1]:
        # Filter the images for the current drift label
        drift_fil = y_preds == drift_label

        # Iterate over each image's importance vector
        for image_imp in sample_local_imp[drift_fil]:
            # Sort the importance vector and get the most important local concept (the index)
            local_imp_sort = np.argsort(image_imp)[::-1][0]

            # Update the respective counter based on drift label
            if drift_label == 0:
                before_drift_counter[local_imp_sort] += 1
            elif drift_label == 1:
                after_drift_counter[local_imp_sort] += 1
        # elif drift_label == 2:
        #     both_drift_counter[local_imp_sort] += 1

    # Calculate total counts for each drift phase
    total_before = sum(before_drift_counter.values())
    total_after = sum(after_drift_counter.values())
    # total_both = sum(both_drift_counter.values())

    # Normalize counts to create probability distributions
    before_drift_dist = {concept: count / total_before for concept, count in before_drift_counter.items()}
    after_drift_dist = {concept: count / total_after for concept, count in after_drift_counter.items()}
    # both_drift_dist = {concept: count / total_both for concept, count in both_drift_counter.items()}

    # Sort concepts by their probability for each distribution
    prob_dict = {0: before_drift_dist, 1: after_drift_dist}
    return prob_dict


def activation_transform_imp_concepts(inputs, concept_args, basis):
    """
    Transforms the input images into an (N, 320) representation where N is the number of images.

    Parameters:
    - inputs: Input images or data to be transformed.
    - patches: Whether to use patches (if needed for some other functionality).
    - labels: Optional labels for the inputs.
    - activations: Optional pre-computed activations. If None, activations are computed.
    - drift_basis: Predefined basis for NMF.
    - n_patches: Number of patches per image (default is 16).

    Returns:
    - transformed_data: Transformed dataset with shape (N, 320).
    """
    # Step 1: Extract latent activations using drift_craft
    A = inputs  # Assuming A.shape = (N, H, W, D) where D is the activation dimension

    # Step 2: Reshape activations to 2D (flatten the spatial dimensions)
    original_shape = A.shape[:-1]  # Keep original shape to reconstruct later
    re_activations = np.reshape(A, (-1, A.shape[-1]))  # Flatten to (N * H * W, D)

    # Step 3: Apply Non-negative Matrix Factorization (NMF) to reduce dimensionality
    embedding, basis, n_iter = non_negative_factorization(np.array(re_activations),
                                                          n_components=len(basis[concept_args]),
                                                          init='custom',
                                                          update_H=False, solver='mu', H=basis[concept_args])

    return embedding


def reconstruct_inputs(inputs, local_importances, basis, num_concepts=1):
    new_acts = []
    for act, local_imp in zip(inputs, local_importances):
        max_local_3 = np.argsort(local_imp)[::-1][:num_concepts]
        nmf_activation = activation_transform_imp_concepts(act, max_local_3, basis)
        act_new = nmf_activation @ basis[max_local_3]
        # nmf_activation = activation_transform(act)
        # act_new = nmf_activation @ drift_basis
        new_acts.append(act_new)
    return np.array(new_acts).reshape((len(new_acts), basis.shape[1]))