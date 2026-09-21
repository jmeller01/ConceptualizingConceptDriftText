from math import ceil
import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import NMF, non_negative_factorization


# return given device choose cuda/cpu automatically if None
def _resolve_device(device):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

# compute g(x) in batches
def batch_inference(model_wrapper, encoded, batch_size, device):
    n = encoded["input_ids"].shape[0]
    results = []
    with torch.no_grad():
        for i in range(ceil(n / batch_size)):
            start, end = i * batch_size, (i + 1) * batch_size
            ids = encoded["input_ids"][start:end].to(device)
            mask = encoded["attention_mask"][start:end].to(device)
            results.append(model_wrapper.features(ids, mask).cpu())
    return torch.cat(results)

# prohject activations on fixed drift_basis
def project_on_basis(activations, basis):
    u, _, _ = non_negative_factorization(
        activations, n_components=len(basis), init="custom", update_H=False, solver="mu", H=basis
    )
    return u

# shared patch extraction and embeddding
class CraftTextBase:
    

    def __init__(self, model_wrapper, tokenizer, num_concepts,
                 patch_mode = "sentence", min_words = 5, max_length = 512,
                 win_size = 15, stride = 10,
                 batch_size = 64, device = None):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.number_of_concepts = num_concepts
        self.patch_mode = patch_mode
        self.min_words = min_words
        self.max_length = max_length
        self.win_size = win_size
        self.stride = stride
        self.batch_size = batch_size
        self.device = _resolve_device(device)

    # split text in patches according to given mode
    def _split_text(self, text):
        if self.patch_mode == "sentence":
            raw_patches = sent_tokenize(text)
        elif self.patch_mode == "window":
            words = text.split()
            raw_patches = [" ".join(words[i:i + self.win_size])
                           for i in range(0, max(1, len(words) - self.win_size + 1), self.stride)]
        else:
            raise ValueError(f"unknown patch mode: {self.patch_mode!r}")

        raw_patches = [p for p in raw_patches if len(p.split()) >= self.min_words]
        if not raw_patches:
            return []

        token_counts = [len(ids) for ids in self.tokenizer(raw_patches, truncation=False)["input_ids"]]
        selected, budget = [], 0
        for patch, n_tokens in zip(raw_patches, token_counts):
            if budget + n_tokens > self.max_length:
                break
            selected.append(patch)
            budget += n_tokens
        return selected

    # embed the patches returns patches its activations, labels and original text idx (n_patxhes can vary per text) 
    def _extract_patches(self, texts, labels):
        crops, patch_labels, text_indices = [], [], []
        for i, (text, label) in enumerate(zip(texts, labels)):
            patches = self._split_text(text)
            crops.extend(patches)
            patch_labels.extend([label] * len(patches))
            text_indices.extend([i] * len(patches))
        if not crops:
            raise ValueError("no patches extracted; check patch_mode/min_words")
        encoded = self.tokenizer(crops, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        activations = batch_inference(self.model_wrapper, encoded, self.batch_size, self.device)
        return crops, activations, patch_labels, text_indices

#concept extraction for one drift phase
class CraftText(CraftTextBase):
    # factorize patch activations into concept bank and coeeficients
    def fit(self, texts, labels):
        crops, activations, patch_labels, text_indices = self._extract_patches(texts, labels)
        activations_np = activations.numpy()

        reducer = NMF(n_components=self.number_of_concepts, alpha_W=1e-2, init="nndsvda", max_iter=1000)
        crops_u = reducer.fit_transform(activations_np)
        concept_bank_w = reducer.components_.astype(np.float32)

        self.crops = crops
        self.activations = activations_np
        self.patch_labels = patch_labels
        self.text_indices = text_indices
        self.U = crops_u
        self.V = concept_bank_w
        return crops, crops_u, concept_bank_w

# projection onto fixed basis from both phases
class CraftTextCombined(CraftTextBase):
    def __init__(self, model_wrapper, tokenizer, basis,
                 patch_mode = "sentence", min_words = 5, max_length = 512,
                 win_size = 15, stride = 10,
                 batch_size = 64, device= None):
        super().__init__(model_wrapper, tokenizer, num_concepts=len(basis), patch_mode=patch_mode,
                         min_words=min_words, max_length=max_length, win_size=win_size,
                         stride=stride, batch_size=batch_size, device=device)
        self.basis = basis
        self.sensitivities = {}   # filled by estimate_importance_helper_l

    # transform patch activations to concept coefficients
    def transform_all(self, texts, labels):
        crops, activations, patch_labels, text_indices = self._extract_patches(texts, labels)
        self.crops = crops
        self.patch_labels = patch_labels
        self.text_indices = text_indices
        self.embedding = project_on_basis(activations.numpy(), self.basis)
        return self.embedding

# embedding of whole texts
def full_text_activations(texts, model_wrapper, tokenizer, batch_size = 64,
                          device= None, max_length = 512):
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return batch_inference(model_wrapper, encoded, batch_size, _resolve_device(device)).numpy()

# determining top n sentences for a given concept
def top_sentences(U, patches_text, concept_id, top_n = 5):
    activations = U[:, concept_id]
    top_idx = np.argsort(activations)[::-1][:top_n]
    return [(patches_text[i], float(activations[i])) for i in top_idx]


# concept coefficients of whole text (row 0) and of every one-word deletion for occlusion
def calculate_u_values(sentence, cropped_words, model_wrapper, tokenizer,
                       basis, separate, ignore_words = None,
                       device = None, max_length= 512,
                       batch_size = 64):
    ignore_words = set(ignore_words or [])
    device = _resolve_device(device)
    variants = [sentence] + [separate.join(np.delete(cropped_words, i))
                             for i, w in enumerate(cropped_words) if w not in ignore_words]
    encoded = tokenizer(variants, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    activations = batch_inference(model_wrapper, encoded, batch_size, device).numpy()
    return project_on_basis(activations, basis)

# occlusion based word importances 
def calculate_importance(words, u_values, concept_id,
                         ignore_words):
    ignore_words = set(ignore_words)
    u_delta = u_values[0, concept_id] - u_values[1:, concept_id]
    importances, delta_id = [], 0
    for word in words:
        if word not in ignore_words:
            importances.append(float(u_delta[delta_id]))
            delta_id += 1
        else:
            importances.append(0.0)
    return importances

# local explanation. for given text: word level importances for each concept
def occlusion_concepts_text(sentence, model_wrapper, tokenizer, basis,
                            concept_ids, ignore_words = None,
                            device= None, max_length= 512):
    ignore_words = list(ignore_words or [])
    sentence = str(sentence)
    words = word_tokenize(sentence)
    u_values = calculate_u_values(sentence, words, model_wrapper, tokenizer, basis, " ",
                                  ignore_words, device, max_length)
    phi = np.array([calculate_importance(words, u_values, c, ignore_words) for c in concept_ids])
    return words, phi

# global explanation. aggregate occlusion values over top n patches that activate given concept most strongly
def occlusion_global_keywords(concept_id, U, patches_text, basis,
                              model_wrapper, tokenizer, ignore_words,
                              top_n_patches =10, top_n_words= 5,
                              device= None, aggregation = "sum"):
    if aggregation not in ("sum", "mean", "norm_sum"):
        raise ValueError(f"unknown aggregation: {aggregation!r}")
    ignore = set(ignore_words)
    scores = {}
    counts = {}
    for patch_text, _ in top_sentences(U, patches_text, concept_id, top_n=top_n_patches):
        words, phi = occlusion_concepts_text(patch_text.lower(), model_wrapper, tokenizer, basis,
                                             concept_ids=[concept_id], ignore_words=ignore, device=device)
        phi = phi[0]
        if aggregation == "norm_sum":
            phi = phi / (np.max(np.abs(phi)) + 1e-5)
        for word, val in zip(words, phi):
            if word in ignore:
                continue
            if aggregation == "mean":
                scores[word] = scores.get(word, 0.0) + val
                counts[word] = counts.get(word, 0) + 1
            elif val > 0:
                scores[word] = scores.get(word, 0.0) + val
    if aggregation == "mean":
        scores = {w: s / counts[w] for w, s in scores.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n_words]
    return dict(ranked)
