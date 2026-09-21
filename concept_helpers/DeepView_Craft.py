from xplique.attributions.global_sensitivity_analysis import HaltonSequenceRS, JansenEstimator
from xplique.concepts.craft import BaseCraft, DisplayImportancesOrder, Factorization, Sensitivity
from xplique.concepts import CraftTorch
from xplique.plots.image import _clip_percentile
from xplique.concepts import CraftManagerTorch
from xplique.concepts.craft_torch import _batch_inference
from typing import Callable, Optional, Tuple
from math import ceil
from matplotlib import gridspec
# from .disc_nmf import RSNMF
# from .shared_disc_nmf import RSSNMF

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CraftTorchDV(CraftTorch):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism on Pytorch.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Pytorch model (torch.nn.modules.module.Module) accepting
        data of shape (n_samples, channels, height, width).
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Pytorch model (torch.nn.modules.module.Module).
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches (crops) to extract from the input data. Default is 64.
    device
        The device to use. Default is 'cuda'.
    """

    def __init__(self, input_to_latent_model: Callable,
                       latent_to_logit_model: Callable,
                       number_of_concepts: int = 20,
                       labels = np.ndarray,
                       batch_size: int = 64,
                       patch_size: int = 64,
                       device : str = 'cuda'):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts, batch_size)
        self.patch_size = patch_size
        self.device = device
        # self.labels = labels
        # self.num_classes = len(np.unique(labels))
        # self.number_of_concepts = number_of_concepts * self.num_classes

        # Check model type
        is_torch_model = issubclass(type(input_to_latent_model), torch.nn.modules.module.Module) & \
                         issubclass(type(latent_to_logit_model), torch.nn.modules.module.Module)
        if not is_torch_model:
            raise TypeError('input_to_latent_model and latent_to_logit_model are not ' \
                            'Pytorch modules')
    def _logit_predict(self, activations: np.ndarray, resize=None) -> torch.Tensor:
        """
        Compute logits from activations using the 2nd model `latent_to_logit_model`.

        Parameters
        ----------
        activations
            Activations produced by the 1st model `input_to_latent_model`,
            of shape (n_samples, height, width, channels).

        Returns
        -------
        logits
            The logits of shape (n_samples, n_classes)
        """
        # pylint disable=no-member
        activations_perturbated = torch.from_numpy(activations.astype(np.float32))

        if len(activations_perturbated.shape) == 4:
            # activations_perturbated: (N, H, W, C) -> (N, C, H, W)
            activations_perturbated = activations_perturbated.permute(0, 3, 1, 2)

        y_pred = _batch_inference(self.latent_to_logit_model, activations_perturbated,
                                  self.batch_size, resize, device=self.device)
        return self._to_np_array(y_pred)

    # def fit(self,
    #         inputs : np.ndarray,
    #         class_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Fit the Craft model to the input data.
    #
    #     Parameters
    #     ----------
    #     inputs
    #         Input data of shape (n_samples, height, width, channels).
    #         (x1, x2, ..., xn) in the paper.
    #     class_id
    #         The class id of the inputs.
    #
    #     Returns
    #     -------
    #     crops
    #         The crops (X in the paper)
    #     crops_u
    #         The concepts' values (U in the paper)
    #     concept_bank_w
    #         The concept's basis (W in the paper)
    #
    #     """
    #     # extract patches (crops) from the input data
    #     crops, activations, labels = self._extract_patches(inputs, self.labels)
    #     # print(activations.shape)
    #     self.new_labels = labels#[[0 if x == 162 else 1] for x in labels.tolist()]
    #
    #     # apply NMF to the activations to obtain matrices U and W
    #     reducer = RSNMF(n_components=self.number_of_concepts,p=2,mu=.005, normalize=True, max_iter=1000)
    #     concept_bank_w, crops_u = reducer.fit_transform(activations.T, y=self.new_labels)
    #     # concept_bank_w = reducer.components_.astype(np.float32)
    #
    #     self.factorization = Factorization(inputs, class_id, crops,
    #                                        reducer, crops_u.T, concept_bank_w.T)
    #
    #     return crops, concept_bank_w, crops_u
    #
    # def transform(self, inputs : np.ndarray, activations : np.ndarray = None) -> np.ndarray:
    #     """Transforms the inputs data into its concept representation.
    #
    #     Parameters
    #     ----------
    #     inputs
    #         The input data to be transformed.
    #     activations
    #         Pre-computed activations of the input data. If not provided, the activations
    #         will be computed using the input_to_latent_model model on the inputs.
    #
    #     Returns
    #     -------
    #     coeffs_u
    #         The concepts' values of the inputs (U in the paper).
    #     """
    #     self.check_if_fitted()
    #
    #     if activations is None:
    #         activations = self._latent_predict(inputs)
    #         # print("activations",activations.shape)
    #
    #     is_4d = len(activations.shape) == 4
    #
    #     if is_4d:
    #         # (N, W, H, C) -> (N * W * H, C)
    #         original_shape = activations.shape[:-1]
    #         re_activations = np.reshape(activations, (-1, activations.shape[-1])).T
    #         # new_shape = activations.shape[:-1]
    #         # print(re_activations.shape)
    #         # n = np.repeat(self.labels, activations.shape[1] * activations.shape[2])
    #         # nl = n#[[0 if x == 162 else 1] for x in n.tolist()]
    #
    #     w_dtype = self.factorization.reducer.components_.dtype
    #     coeffs_u = self.factorization.reducer.transform(self._to_np_array(re_activations,
    #                                                             dtype=w_dtype))
    #     # print(coeffs_u.shape)
    #     coeffs_u = coeffs_u.T
    #     # print(coeffs_u.shape)
    #     if is_4d:
    #         # (N * W * H, R) -> (N, W, H, R) with R = nb_concepts
    #         coeffs_u = np.reshape(coeffs_u, (*original_shape, coeffs_u.shape[-1]))
    #         # print(coeffs_u.shape)
    #     return coeffs_u
    #
    # def _extract_patches(self, inputs: torch.Tensor, labels: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    #     """
    #     Extract patches (crops) from the input images, and compute their embeddings.
    #
    #     Parameters
    #     ----------
    #     inputs
    #         Input images (n_samples, channels, height, width)
    #     labels
    #         Labels for the input images (n_samples,)
    #
    #     Returns
    #     -------
    #     patches
    #         A tuple containing the patches (n_patches, channels, height, width).
    #     activations
    #         The patches activations (n_patches, channels).
    #     patch_labels
    #         The labels associated with each patch (n_patches,).
    #     """
    #
    #     image_size = (inputs.shape[2], inputs.shape[3])
    #     num_channels = inputs.shape[1]
    #
    #     # Extract patches from the input data
    #     strides = int(self.patch_size * 0.80)
    #     patches = torch.nn.functional.unfold(inputs, kernel_size=self.patch_size, stride=strides)
    #     patches = patches.transpose(1, 2).contiguous().view(-1, num_channels, self.patch_size, self.patch_size)
    #
    #     # Compute the activations of the patches
    #     activations = self._latent_predict(patches, resize=image_size)
    #
    #     assert torch.min(activations) >= 0.0, "Activations must be positive."
    #
    #     if len(activations.shape) == 4:
    #         activations = torch.mean(activations, dim=(1, 2))
    #
    #     height, width = image_size
    #     num_patches_per_image = ((height - self.patch_size) // strides + 1) * ((width - self.patch_size) // strides + 1)
    #
    #     # Create a numpy array to store the label of each patch
    #     patch_labels = np.repeat(labels, num_patches_per_image)
    #
    #     # Check for consistency
    #     if len(patch_labels) != patches.shape[0]:
    #         raise ValueError(f"Mismatch: {len(patch_labels)} patch labels, but {patches.shape[0]} patches found.")
    #
    #     return self._to_np_array(patches), self._to_np_array(activations), patch_labels
    #
    # def estimate_importance(self, inputs : np.ndarray = None, nb_design: int = 32) -> np.ndarray:
    #     """
    #     Estimates the importance of each concept for a given class, either globally
    #     on the whole dataset provided in the fit() method (in this case, inputs shall
    #     be set to None), or locally on a specific input image.
    #
    #     Parameters
    #     ----------
    #     inputs : numpy array or Tensor
    #         The input data on which to compute the importances.
    #         If None, then the inputs provided in the fit() method
    #         will be used (global importance of the whole dataset).
    #         Default is None.
    #     nb_design
    #         The number of design to use for the importance estimation. Default is 32.
    #
    #     Returns
    #     -------
    #     importances
    #         The Sobol total index (importance score) for each concept.
    #
    #     """
    #     self.check_if_fitted()
    #
    #     compute_global_importances = False
    #     if inputs is None:
    #         inputs = self.factorization.inputs
    #         compute_global_importances = True
    #
    #     coeffs_u = self.transform(inputs)
    #
    #
    #     masks = HaltonSequenceRS()(self.number_of_concepts, nb_design = nb_design)
    #     estimator = JansenEstimator()
    #
    #     importances = []
    #
    #     if len(coeffs_u.shape) == 2:
    #         # apply the original method of the paper
    #
    #         for coeff in coeffs_u:
    #             u_perturbated = coeff[None, :] * masks
    #             a_perturbated = u_perturbated @ self.factorization.concept_bank_w
    #
    #             y_pred = self._logit_predict(a_perturbated)
    #             y_pred = y_pred[:, self.factorization.class_id]
    #
    #             stis = estimator(masks, y_pred, nb_design)
    #
    #             importances.append(stis)
    #
    #     elif len(coeffs_u.shape) == 4:
    #         # apply a re-parameterization trick and use mask on all localization for a given
    #         # concept id to estimate sobol indices
    #         for coeff in coeffs_u:
    #
    #             u_perturbated = masks[:, None, None, :] * coeff[None, :]
    #
    #             a_perturbated =  np.reshape(u_perturbated,
    #                                     (-1, coeff.shape[-1])) @ self.factorization.concept_bank_w
    #             # print("a_perturbed", a_perturbated.shape)
    #             a_perturbated = np.reshape(a_perturbated,
    #                                     (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))
    #             # print("a_perturbed-re", a_perturbated.shape)
    #
    #             # a_perturbated: (N, H, W, C)
    #             y_pred = self._logit_predict(a_perturbated)
    #             # print("preds",y_pred.shape)
    #
    #             y_pred = y_pred[:, self.factorization.class_id]
    #             # print("preds",y_pred.shape)
    #             # print("preds_shape", y_pred.shape)
    #
    #             stis = estimator(masks, y_pred, nb_design)
    #
    #             importances.append(stis)
    #
    #     importances = np.mean(importances, 0)
    #
    #     # Save the results of the computation if working on the whole dataset
    #     if compute_global_importances:
    #         most_important_concepts = np.argsort(importances)[::-1]
    #         self.sensitivity = Sensitivity(importances, most_important_concepts)
    #
    #     return importances
    def plot_image_concepts(self,
                            img: np.ndarray,
                            img_local_importance: np.ndarray,
                            yt: int,
                            yp:int,
                            display_importance_order: DisplayImportancesOrder = \
                                    DisplayImportancesOrder.GLOBAL,
                            nb_most_important_concepts: int = 5,
                            filter_percentile: int = 90,
                            clip_percentile: Optional[float] = 10,
                            alpha: float = 0.65,
                            filepath: Optional[str] = None,
                            **plot_kwargs):
        """
        All in one method displaying several plots for the image `id` given in argument:
        - the concepts attribution map for this image
        - the best crops for each concept (displayed around the heatmap)
        - the importance of each concept

        Parameters
        ----------
        img
            The image to display.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
            Default to GLOBAL.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        filepath
            Path the file will be saved at. If None, the function will call plt.show().
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        fig = plt.figure(figsize=(18, 7))

        if display_importance_order == DisplayImportancesOrder.LOCAL:
            # compute the importances for the sample input in argument
            importances = self.estimate_importance(inputs=img)
            most_important_concepts = np.argsort(importances)[::-1][:nb_most_important_concepts]
        else:
            # use the global importances computed on the whole dataset
            importances = self.sensitivity.importances
            most_important_concepts = \
                self.sensitivity.most_important_concepts[:nb_most_important_concepts]

        # create the main gridspec which is split in the left and right parts storing
        # the crops, and the central part to display the heatmap
        nb_rows = ceil(len(most_important_concepts) / 2.0)
        nb_cols = 4
        gs_main = fig.add_gridspec(nb_rows, nb_cols, hspace=0.4, width_ratios=[0.2, 0.4, 0.2, 0.4])

        # Add ghost axes and titles on gs1 and gs2
        ax = fig.add_subplot(gs_main[:])

        ax.set_title('True Label:' + str(yt) + '\nPredicted Label:' + str(yp))
        # Central image
        #
        fig.add_subplot(gs_main[:, 1])
        self.plot_concept_attribution_map(image=img,
                                          most_important_concepts=most_important_concepts,
                                          nb_most_important_concepts=nb_most_important_concepts,
                                          filter_percentile=filter_percentile,
                                          clip_percentile=clip_percentile,
                                          alpha=alpha,
                                          **plot_kwargs)

        fig.add_subplot(gs_main[2, 1])
        self.plot_concepts_importances(importances=img_local_importance,
                                       display_importance_order=DisplayImportancesOrder.LOCAL,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        # Concepts: creation of the axes on left and right of the image for the concepts
        #
        gs_concepts_axes = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 0])
                            for i in range(nb_rows)]
        gs_right = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 2])
                    for i in range(nb_rows)]
        gs_concepts_axes.extend(gs_right)

        # display the best crops for each concept, in the order of the most important concept
        nb_crops = 6

        # compute the right color to use for the crops
        global_color_index_order = np.argsort(self.sensitivity.importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        local_cmap = np.array(self.sensitivity.cmaps)[local_color_index_order]

        for i, c_id in enumerate(most_important_concepts):
            cmap = local_cmap[i]

            # use a ghost invisible subplot only to have a border around the crops
            ghost_axe = fig.add_subplot(gs_concepts_axes[i][:, :])
            ghost_axe.set_title(f"{c_id}", color=cmap(1.0))
            ghost_axe.axis('off')

            inset_axes = ghost_axe.inset_axes([-0.04, -0.04, 1.08, 1.08])  # outer border
            inset_axes.set_xticks([])
            inset_axes.set_yticks([])
            for spine in inset_axes.spines.values():  # border color
                spine.set_edgecolor(color=cmap(1.0))
                spine.set_linewidth(3)

            # draw each crop for this concept
            gs_current = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=
            gs_concepts_axes[i][:, :])

            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]
            for i in range(nb_crops):
                axe = plt.Subplot(fig, gs_current[i // 3, i % 3])
                fig.add_subplot(axe)
                BaseCraft._show(best_crops[i])

        # Right plot: importances
        importance_axe = gridspec.GridSpecFromSubplotSpec(3, 2, width_ratios=[0.1, 0.9],
                                                          height_ratios=[0.15, 0.6, 0.15],
                                                          subplot_spec=gs_main[:, 3])
        fig.add_subplot(importance_axe[1, 1])
        self.plot_concepts_importances(importances=importances,
                                       display_importance_order=display_importance_order,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)



        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

class CraftTorchSupervised(CraftTorch):
    """
    Class implementing the CraftManager on Tensorflow.
    This manager creates one CraftTorch instance per class to explain.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must return positive activations.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    inputs
        Input data of shape (n_samples, height, width, channels).
        (x1, x2, ..., xn) in the paper.
    labels
        Labels of the inputs of shape (n_samples, class_id)
    list_of_class_of_interest
        A list of the classes id to explain. The manager will instanciate one
        CraftTorch object per element of this list.
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches (crops) to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 number_of_concepts: int = 20,
                 inputs = np.ndarray,
                 labels = np.ndarray,
                 batch_size: int = 64,
                 patch_size: int = 64,
                 device: str = 'cuda'):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts, batch_size)

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.device = device
        self.inputs = inputs
        self.labels =  labels
        self.new_labels = None
        self.classes_of_interest = np.unique(labels)
        self.num_classes = len(np.unique(labels))
        self.components_per_class = number_of_concepts
        self.number_of_concepts = self.components_per_class * self.num_classes
        self.craft_instances = {}
        self.sensitivities = {}
        self.transform_num_data_dict = {}
        # self.transform_I
        # for class_of_interest in self.list_of_class_of_interest:
        #     craft = CraftTorch(input_to_latent_model, latent_to_logit_model,
        #                        number_of_concepts, batch_size, patch_size, device)
        #     self.craft_instances[class_of_interest] = craft

    def compute_predictions(self):
        """
        Compute the predictions for the current dataset, using the 2 models
        input_to_latent_model and latent_to_logit_model chained.

        Returns
        -------
        y_preds
            the predictions
        """
        model = nn.Sequential(self.input_to_latent_model, self.latent_to_logit_model)
        logit_activations = _batch_inference(model, self.inputs, self.batch_size, None,
                                       device=self.device)
        y_preds = np.array(torch.argmax(logit_activations, -1))  # pylint disable=no-member

        activations = self._latent_predict(self.inputs)

        return y_preds, activations

    def _logit_predict(self, activations: np.ndarray, resize=None) -> torch.Tensor:
        """
        Compute logits from activations using the 2nd model `latent_to_logit_model`.

        Parameters
        ----------
        activations
            Activations produced by the 1st model `input_to_latent_model`,
            of shape (n_samples, height, width, channels).

        Returns
        -------
        logits
            The logits of shape (n_samples, n_classes)
        """
        # pylint disable=no-member
        activations_perturbated = torch.from_numpy(activations.astype(np.float32))

        if len(activations_perturbated.shape) == 4:
        #     # activations_perturbated: (N, H, W, C) -> (N, C, H, W)
        #     activations_perturbated = activations_perturbated.permute(0, 2, 3, 1)
            activations_perturbated = activations_perturbated.permute(0, 3, 1, 2)

        y_pred = _batch_inference(self.latent_to_logit_model, activations_perturbated,
                                  self.batch_size, resize, device=self.device)
        return self._to_np_array(y_pred)

    def init_I(self, m, num_data, labels, classes, num_data_class):
        u_labels = np.unique(labels)
        num_classes = len(classes)
        matrix = np.ones((m * num_classes, num_data))
        current_col_index = 0

        if len(u_labels) > 1:
            for i in range(num_classes):
                start_row_index = i * m
                end_row_index = start_row_index + m

                num_points_in_class = num_data_class[u_labels[i]]

                matrix[start_row_index:end_row_index, current_col_index:current_col_index + num_points_in_class] = 0

                current_col_index += num_points_in_class

        elif len(u_labels) == 1 and len(num_data_class.keys()) == 1:
            matrix = np.zeros((m * num_classes, num_data))

        else:
            for i, label in enumerate(classes):
                start_row_index = label * m
                end_row_index = start_row_index + m

                if label in u_labels:
                    num_points_in_class = num_data_class[classes[i]]
                    matrix[start_row_index:end_row_index, current_col_index:current_col_index + num_points_in_class] = 0
                    current_col_index += num_points_in_class

                current_col_index += num_data_class[classes[i]]

        return matrix
    def _extract_patches(self, inputs: torch.Tensor, labels: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Extract patches (crops) from the input images, and compute their embeddings.

        Parameters
        ----------
        inputs
            Input images (n_samples, channels, height, width)
        labels
            Labels for the input images (n_samples,)

        Returns
        -------
        patches
            A tuple containing the patches (n_patches, channels, height, width).
        activations
            The patches activations (n_patches, channels).
        patch_labels
            The labels associated with each patch (n_patches,).
        """

        image_size = (inputs.shape[2], inputs.shape[3])
        num_channels = inputs.shape[1]

        # Extract patches from the input data
        strides = int(self.patch_size * 0.80)
        patches = torch.nn.functional.unfold(inputs, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, num_channels, self.patch_size, self.patch_size)

        # Compute the activations of the patches
        activations = self._latent_predict(patches, resize=image_size)
        # print(activations.shape)

        # assert torch.min(activations) >= 0.0, "Activations must be positive."

        if len(activations.shape) == 4:
            activations = torch.mean(activations, dim=(1, 2))
            # original_shape = activations.shape[:-1]
            # activations = np.reshape(activations, (-1, activations.shape[-1]))
            # print(activations.shape)

        height, width = image_size
        num_patches_per_image = ((height - self.patch_size) // strides + 1) * ((width - self.patch_size) // strides + 1)
        #
        # # Create a numpy array to store the label of each patch
        patch_labels = np.repeat(labels, num_patches_per_image)
        #
        # # Check for consistency
        # if len(patch_labels) != patches.shape[0]:
        #     raise ValueError(f"Mismatch: {len(patch_labels)} patch labels, but {patches.shape[0]} patches found.")

        return self._to_np_array(patches), self._to_np_array(activations), patch_labels

    def fit(self,
            class_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, height, width, channels).
            (x1, x2, ..., xn) in the paper.
        class_id
            The class id of the inputs.

        Returns
        -------
        crops
            The crops (X in the paper)
        crops_u
            The concepts' values (U in the paper)
        concept_bank_w
            The concept's basis (W in the paper)

        """
        # extract patches (crops) from the input data
        crops, activations, labels = self._extract_patches(self.inputs, self.labels)
        #
        # print(activations.T.shape)
        # print(activations.shape)
        self.new_labels = labels
        num_data_class = {label: len(np.where(labels == label)[0]) for label in self.classes_of_interest}
        fit_I = self.init_I(self.components_per_class, activations.shape[0],labels,self.classes_of_interest,num_data_class=num_data_class)

        # apply NMF to the activations to obtain matrices U and W
        reducer = RSNMF(n_components=self.number_of_concepts, component_per_class=self.components_per_class, p=.5,mu=0.1, normalize=True, max_iter=1000)
        concept_bank_w, crops_u = reducer.fit_transform(activations.T, y=fit_I)
        # concept_bank_w = reducer.components_.astype(np.float32)

        self.factorization = Factorization(self.inputs, class_id, crops,
                                           reducer, crops_u.T, concept_bank_w.T)

        return crops, concept_bank_w, crops_u

    def transform(self, inputs : np.ndarray, labels : np.ndarray, activations : np.ndarray = None) -> np.ndarray:
        """Transforms the inputs data into its concept representation.

        Parameters
        ----------
        inputs
            The input data to be transformed.
        activations
            Pre-computed activations of the input data. If not provided, the activations
            will be computed using the input_to_latent_model model on the inputs.

        Returns
        -------
        coeffs_u
            The concepts' values of the inputs (U in the paper).
        """
        self.check_if_fitted()

        if activations is None:
            activations = self._latent_predict(inputs)
            # print("activations",activations.shape)

        is_4d = len(activations.shape) == 4
        #
        if is_4d:
            # (N, W, H, C) -> (N * W * H, C)
            original_shape = activations.shape[:-1]
            re_activations = np.reshape(activations, (-1, activations.shape[-1]))
            # print(re_activations.shape)
            # new_shape = activations.shape[:-1]
            # print(re_activations.shape)
            # n = np.repeat(labels, activations.shape[1] * activations.shape[2])
            # transform_num_data_dict = {label: len(np.where(n == label)[0]) for label in self.classes_of_interest})
            # transform_I = self.init_I(self.components_per_class, re_activations.shape[0], n, self.classes_of_interest,
            #                                 num_data_class=transform_num_data_dict)

            # nl = n#[[0 if x == 162 else 1] for x in n.tolist()]
        # y = (self.transform_I if activations is not None else None )
        n = np.repeat(labels, activations.shape[1] * activations.shape[2])
        transform_num_data_dict = {label: len(np.where(n == label)[0])
                                   for label in np.unique(labels)}
        y = self.init_I(self.components_per_class, re_activations.shape[0], n, self.classes_of_interest,
                                      num_data_class=transform_num_data_dict)
        # print(y)
        w_dtype = self.factorization.reducer.components_.dtype
        coeffs_u = self.factorization.reducer.transform(self._to_np_array(re_activations,
                                                                dtype=w_dtype).T, y=y)

        coeffs_u = coeffs_u.T

        if is_4d:
            # (N * W * H, R) -> (N, W, H, R) with R = nb_concepts
            coeffs_u = np.reshape(coeffs_u, (*original_shape, coeffs_u.shape[-1]))
            # print(coeffs_u.shape)
        return coeffs_u

    def estimate_importance(self, nb_design: int = 32, verbose: bool = False):
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
        y_preds, activations = self.compute_predictions()
        # print(activations.shape)
        is_4d = len(activations.shape) == 4

        # if is_4d:
        #     # (N, W, H, C) -> (N * W * H, C)
        #     original_shape = activations.shape[:-1]
        #     re_activations = np.reshape(activations, (-1, activations.shape[-1]))
        #     # new_shape = activations.shape[:-1]
        #     # print(re_activations.shape)
        #     n = np.repeat(self.labels, activations.shape[1] * activations.shape[2])
        #     self.transform_num_data_dict = {label: len(np.where(n == label)[0]) for label in self.classes_of_interest}
        #     self.transform_I = self.init_I(self.components_per_class, re_activations.shape[0], n, self.classes_of_interest,
        #                                   num_data_class=self.transform_num_data_dict)

        global_importance = []
        for class_of_interest in self.classes_of_interest:
            filtered_indices = np.where(y_preds == class_of_interest)
            class_inputs = self.inputs[filtered_indices]
            class_labels = self.labels[filtered_indices]
            class_activations = activations[filtered_indices]
            class_preds = y_preds[filtered_indices]
            # if nb_samples_per_class is not None:
            #     class_inputs = class_inputs[:nb_samples_per_class]
            #     class_labels = class_labels[:nb_samples_per_class]
            importances = self.estimate_importance_helper(inputs=class_inputs,
                                                          labels=class_labels,
                                                          activations = class_activations,
                                                          class_of_interest=class_of_interest,
                                                          nb_design=nb_design,
                                                          compute_class_importance=True)
            global_importance.append(importances)

        return global_importance

    def estimate_importance_helper(self, inputs : np.ndarray = None, labels : np.ndarray = None,
                                   activations : np.ndarray = None, class_of_interest: int = None,
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
        self.check_if_fitted()

        # print(labels)
        coeffs_u = self.transform(inputs, labels, activations)
        # print(coeffs_u.shape)


        masks = HaltonSequenceRS()(self.number_of_concepts, nb_design = nb_design)
        estimator = JansenEstimator()

        importances = []

        if len(coeffs_u.shape) == 2:
            # apply the original method of the paper

            for coeff in coeffs_u:
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.factorization.concept_bank_w

                y_pred = self._logit_predict(a_perturbated)
                y_pred = y_pred[:, class_of_interest]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        elif len(coeffs_u.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for coeff in coeffs_u:

                u_perturbated = masks[:, None, None, :] * coeff[None, :]
                # print(u_perturbated.shape)
                a_perturbated =  np.reshape(u_perturbated,
                                        (-1, coeff.shape[-1])) @ self.factorization.concept_bank_w
                # print("a_perturbed", a_perturbated.shape)
                a_perturbated = np.reshape(a_perturbated,
                                        (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))
                # print("a_perturbed-re", a_perturbated.shape)

                # a_perturbated: (N, H, W, C)
                y_pred = self._logit_predict(a_perturbated)
                # print("preds",y_pred.shape)

                y_pred = y_pred[:, class_of_interest]
                # print("preds",y_pred.shape)
                # print("preds_shape", y_pred.shape)

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        importances = np.mean(importances, 0)

        # Save the results of the computation if working on the whole dataset
        if compute_class_importance:
            most_important_concepts = np.argsort(importances)[::-1]
            self.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts)

        return importances


    def plot_image_concepts(self,
                            img: np.ndarray,
                            img_local_importance: np.ndarray,
                            yt: int,
                            yp: int,
                            display_importance_order: DisplayImportancesOrder = \
                                    DisplayImportancesOrder.GLOBAL,
                            nb_most_important_concepts: int = 5,
                            filter_percentile: int = 90,
                            clip_percentile: Optional[float] = 10,
                            alpha: float = 0.65,
                            filepath: Optional[str] = None,
                            **plot_kwargs):
        """
        All in one method displaying several plots for the image `id` given in argument:
        - the concepts attribution map for this image
        - the best crops for each concept (displayed around the heatmap)
        - the importance of each concept

        Parameters
        ----------
        img
            The image to display.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
            Default to GLOBAL.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        filepath
            Path the file will be saved at. If None, the function will call plt.show().
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        fig = plt.figure(figsize=(18, 7))

        if display_importance_order == DisplayImportancesOrder.LOCAL:
            # compute the importances for the sample input in argument
            importances = self.estimate_importance(inputs=img)
            most_important_concepts = np.argsort(importances)[::-1][:nb_most_important_concepts]
        else:
            # use the global importances computed on the whole dataset
            importances = self.sensitivities[yp].importances
            most_important_concepts = \
                self.sensitivities[yp].most_important_concepts[:nb_most_important_concepts]

        # create the main gridspec which is split in the left and right parts storing
        # the crops, and the central part to display the heatmap
        nb_rows = ceil(len(most_important_concepts) / 2.0)
        nb_cols = 4
        gs_main = fig.add_gridspec(nb_rows, nb_cols, hspace=0.4, width_ratios=[0.2, 0.4, 0.2, 0.4])

        # Add ghost axes and titles on gs1 and gs2
        ax = fig.add_subplot(gs_main[:])

        ax.set_title('True Label:' + str(yt) + '\nPredicted Label:' + str(yp))
        # Central image
        #
        fig.add_subplot(gs_main[:, 1])
        self.plot_concept_attribution_map(image=img,
                                          most_important_concepts=most_important_concepts,
                                          class_concepts=yp,
                                          nb_most_important_concepts=nb_most_important_concepts,
                                          filter_percentile=filter_percentile,
                                          clip_percentile=clip_percentile,
                                          alpha=alpha,
                                          **plot_kwargs)

        fig.add_subplot(gs_main[2, 1])
        self.plot_concepts_importances(class_concepts=yp,
                                       importances=img_local_importance,
                                       display_importance_order=DisplayImportancesOrder.LOCAL,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        # Concepts: creation of the axes on left and right of the image for the concepts
        #
        gs_concepts_axes = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 0])
                            for i in range(nb_rows)]
        gs_right = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 2])
                    for i in range(nb_rows)]
        gs_concepts_axes.extend(gs_right)

        # display the best crops for each concept, in the order of the most important concept
        nb_crops = 6

        # compute the right color to use for the crops
        global_color_index_order = np.argsort(self.sensitivities[yp].importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        local_cmap = np.array(self.sensitivities[yp].cmaps)[local_color_index_order]

        for i, c_id in enumerate(most_important_concepts):
            cmap = local_cmap[i]

            # use a ghost invisible subplot only to have a border around the crops
            ghost_axe = fig.add_subplot(gs_concepts_axes[i][:, :])
            ghost_axe.set_title(f"{c_id}", color=cmap(1.0))
            ghost_axe.axis('off')

            inset_axes = ghost_axe.inset_axes([-0.04, -0.04, 1.08, 1.08])  # outer border
            inset_axes.set_xticks([])
            inset_axes.set_yticks([])
            for spine in inset_axes.spines.values():  # border color
                spine.set_edgecolor(color=cmap(1.0))
                spine.set_linewidth(3)

            # draw each crop for this concept
            gs_current = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=
            gs_concepts_axes[i][:, :])

            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]
            for i in range(nb_crops):
                axe = plt.Subplot(fig, gs_current[i // 3, i % 3])
                fig.add_subplot(axe)
                BaseCraft._show(best_crops[i])

        # Right plot: importances
        importance_axe = gridspec.GridSpecFromSubplotSpec(3, 2, width_ratios=[0.1, 0.9],
                                                          height_ratios=[0.15, 0.6, 0.15],
                                                          subplot_spec=gs_main[:, 3])
        fig.add_subplot(importance_axe[1, 1])
        self.plot_concepts_importances(class_concepts=yp,
                                       importances=importances,
                                       display_importance_order=display_importance_order,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

    def plot_concept_attribution_map(self,
                                     image: np.ndarray,
                                     most_important_concepts: np.ndarray,
                                     class_concepts: int,
                                     nb_most_important_concepts: int = 5,
                                     filter_percentile: int = 90,
                                     clip_percentile: Optional[float] = 10,
                                     alpha: float = 0.65,
                                     **plot_kwargs):
        """
        Display the concepts attribution map for a single image given in argument.

        Parameters
        ----------
        image
            The image to display.
        most_important_concepts
            The concepts ids to display.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap.
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            It is applied after the filter_percentile operation.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        # pylint: disable=E1101
        most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        # Find the colors corresponding to the importances
        global_color_index_order = np.argsort(self.sensitivities[class_concepts].importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        local_cmap = np.array(self.sensitivities[class_concepts].cmaps)[local_color_index_order]

        if image.shape[0] == 3:
            dsize = image.shape[1:3]  # pytorch
        else:
            dsize = image.shape[0:2]  # tf
        BaseCraft._show(image, **plot_kwargs)

        image_u = self.transform(image, labels=class_concepts)[0]
        for i, c_id in enumerate(most_important_concepts[::-1]):
            heatmap = image_u[:, :, c_id]

            # only show concept if excess N-th percentile
            sigma = np.percentile(np.array(heatmap).flatten(), filter_percentile)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            # resize the heatmap before cliping
            heatmap = cv2.resize(heatmap[:, :, None], dsize=dsize,
                                 interpolation=cv2.INTER_CUBIC)
            if clip_percentile:
                heatmap = _clip_percentile(heatmap, clip_percentile)

            BaseCraft._show(heatmap, cmap=local_cmap[::-1][i], alpha=alpha, **plot_kwargs)

    def plot_concepts_importances(self,
                                  class_concepts: int,
                                  importances: np.ndarray = None,
                                  display_importance_order: DisplayImportancesOrder = \
                                          DisplayImportancesOrder.GLOBAL,
                                  nb_most_important_concepts: int = None,
                                  verbose: bool = False):
        """
        Plot a bar chart displaying the importance value of each concept.

        Parameters
        ----------
        importances
            The importances computed by the estimate_importance() method.
            Default is None, in this case the importances computed on the whole
            dataset will be used.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
        nb_most_important_concepts
            The number of concepts to display. If None is provided, then all the concepts
            will be displayed unordered, otherwise only nb_most_important_concepts will be
            displayed, ordered by importance.
            Default is None.
        verbose
            If True, then print the importance value of each concept, otherwise no textual
            output will be printed.
        """

        if importances is None:
            # global
            importances = self.sensitivities[class_concepts].importances
            most_important_concepts = self.sensitivities[class_concepts].most_important_concepts
        else:
            # local
            most_important_concepts = np.argsort(importances)[::-1]

        if nb_most_important_concepts is None:
            # display all concepts not ordered
            global_color_index_order = self.sensitivities[class_concepts].most_important_concepts
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in range(len(importances))]
            colors = np.array([colors(1.0)
                               for colors in self.sensitivities[class_concepts].cmaps])[local_color_index_order]

            plt.bar(range(len(importances)), importances, color=colors)
            plt.xticks(range(len(importances)))

        else:
            # only display the nb_most_important_concepts concepts in their importance order
            most_important_concepts = most_important_concepts[:nb_most_important_concepts]

            # Find the correct color index
            global_color_index_order = np.argsort(self.sensitivities[class_concepts].importances)[::-1]
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in most_important_concepts]
            colors = np.array([colors(1.0)
                               for colors in self.sensitivities[class_concepts].cmaps])[local_color_index_order]

            plt.bar(range(len(importances[most_important_concepts])),
                    importances[most_important_concepts], color=colors)
            plt.xticks(ticks=range(len(most_important_concepts)),
                       labels=most_important_concepts)

        if display_importance_order == DisplayImportancesOrder.GLOBAL:
            importance_order = "Global"
        else:
            importance_order = "Local"
        plt.title(f"{importance_order} Concept Importance")

        if verbose:
            for c_id in most_important_concepts:
                print(f"Concept {c_id} has an importance value of {importances[c_id]:.2f}")

class CraftTorchSupervisedShared(CraftTorchSupervised):
    def __init__(self, input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 number_of_concepts: int = 20,
                 disc_concepts = int,
                 shared_concepts=int,
                 inputs = np.ndarray,
                 labels = np.ndarray,
                 batch_size: int = 64,
                 patch_size: int = 64,
                 device: str = 'cuda'):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts,inputs,labels,batch_size)

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.device = device
        self.inputs = inputs
        self.labels =  labels
        self.new_labels = None
        self.classes_of_interest = np.unique(labels)
        self.num_classes = len(np.unique(labels))
        self.disc_concepts = disc_concepts
        self.shared_concepts = shared_concepts
        self.number_of_concepts = (self.disc_concepts * self.num_classes) + self.shared_concepts
        self.craft_instances = {}
        self.sensitivities = {}
        self.transform_num_data_dict = {}

    def init_I(self, m, shared_components, num_data, labels, classes, num_data_class):
        u_labels = np.unique(labels)
        num_classes = len(classes)
        matrix = np.ones((m * num_classes, num_data))
        shared = np.zeros((shared_components, num_data))
        matrix = np.concatenate((matrix, shared))
        current_col_index = 0

        if len(u_labels) > 1:
            for i in range(num_classes):
                start_row_index = i * m
                end_row_index = start_row_index + m

                num_points_in_class = num_data_class[u_labels[i]]

                matrix[start_row_index:end_row_index, current_col_index:current_col_index + num_points_in_class] = 0

                current_col_index += num_points_in_class

        elif len(u_labels) == 1 and len(num_data_class.keys()) == 1:
            matrix = np.zeros(((m * num_classes) + shared_components, num_data))

        else:
            for i, label in enumerate(classes):
                start_row_index = label * m
                end_row_index = start_row_index + m

                if label in u_labels:
                    num_points_in_class = num_data_class[classes[i]]
                    matrix[start_row_index:end_row_index, current_col_index:current_col_index + num_points_in_class] = 0
                    current_col_index += num_points_in_class

                current_col_index += num_data_class[classes[i]]

        return matrix
    def fit(self,
            class_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, height, width, channels).
            (x1, x2, ..., xn) in the paper.
        class_id
            The class id of the inputs.

        Returns
        -------
        crops
            The crops (X in the paper)
        crops_u
            The concepts' values (U in the paper)
        concept_bank_w
            The concept's basis (W in the paper)

        """
        # extract patches (crops) from the input data
        crops, activations, labels = self._extract_patches(self.inputs, self.labels)
        #
        # print(activations.T.shape)
        # print(activations.shape)
        self.new_labels = labels
        num_data_class = {label: len(np.where(labels == label)[0]) for label in self.classes_of_interest}
        fit_I = self.init_I(self.disc_concepts, self.shared_concepts, activations.shape[0],labels,self.classes_of_interest,num_data_class=num_data_class)

        # apply NMF to the activations to obtain matrices U and W
        reducer = RSSNMF(n_components=self.number_of_concepts, p=.5, mu=0.01, normalize=True, max_iter=1000)
        concept_bank_w, crops_u = reducer.fit_transform(activations.T, y=fit_I)
        # concept_bank_w = reducer.components_.astype(np.float32)

        self.factorization = Factorization(self.inputs, class_id, crops,
                                           reducer, crops_u.T, concept_bank_w.T)

        return crops, concept_bank_w, crops_u

    def transform(self, inputs : np.ndarray, labels:np.ndarray, activations : np.ndarray = None) -> np.ndarray:
        """Transforms the inputs data into its concept representation.

        Parameters
        ----------
        inputs
            The input data to be transformed.
        activations
            Pre-computed activations of the input data. If not provided, the activations
            will be computed using the input_to_latent_model model on the inputs.

        Returns
        -------
        coeffs_u
            The concepts' values of the inputs (U in the paper).
        """
        self.check_if_fitted()

        if activations is None:
            activations = self._latent_predict(inputs)
            # print("activations",activations.shape)

        is_4d = len(activations.shape) == 4
        #
        if is_4d:
            # (N, W, H, C) -> (N * W * H, C)
            original_shape = activations.shape[:-1]
            re_activations = np.reshape(activations, (-1, activations.shape[-1]))
            # print(re_activations.shape)
            # new_shape = activations.shape[:-1]
            # print(re_activations.shape)
            # n = np.repeat(labels, activations.shape[1] * activations.shape[2])
            # transform_num_data_dict = {label: len(np.where(n == label)[0]) for label in self.classes_of_interest})
            # transform_I = self.init_I(self.components_per_class, re_activations.shape[0], n, self.classes_of_interest,
            #                                 num_data_class=transform_num_data_dict)

            # nl = n#[[0 if x == 162 else 1] for x in n.tolist()]
        # y = (self.transform_I if activations is not None else None )
        n = np.repeat(labels, activations.shape[1] * activations.shape[2])
        transform_num_data_dict = {label: len(np.where(n == label)[0])
                                   for label in np.unique(labels)}
        y = self.init_I(self.disc_concepts, self.shared_concepts, re_activations.shape[0], n, self.classes_of_interest,
                                      num_data_class=transform_num_data_dict)
        # print(y)
        w_dtype = self.factorization.reducer.components_.dtype
        coeffs_u = self.factorization.reducer.transform(self._to_np_array(re_activations,
                                                                dtype=w_dtype).T, y=y)

        coeffs_u = coeffs_u.T

        if is_4d:
            # (N * W * H, R) -> (N, W, H, R) with R = nb_concepts
            coeffs_u = np.reshape(coeffs_u, (*original_shape, coeffs_u.shape[-1]))
            # print(coeffs_u.shape)
        return coeffs_u
