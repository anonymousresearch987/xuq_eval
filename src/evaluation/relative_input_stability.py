import numpy as np
import torch
import warnings
from tqdm import tqdm

from src.evaluation.metric import Metric
from src.uncertainty_attributions.empirical_xuq import EmpiricalXUQGenerator
from src.models.models import NeuralNetworkBase


class RelativeInputStability(Metric):
    """The RelativeInputStability class includes functions to assess the continuity of uncertainty attributions.
    Implementation of relative input stability metric by Agarwal et al., 2022.

    References:
        Agarwal, C., Johnson, N., Pawelczyk, M., Krishna, S., Saxena, E., Zitnik, M., & Lakkaraju, H. (2022). Rethinking Stability for Attribution-based Explanations (arXiv:2203.06877). arXiv. https://doi.org/10.48550/arXiv.2203.06877
    """

    name = "RelativeInputStability"
    eps = 1e-6
    noise_std: float = 0.05
    num_perturbations: int

    def __init__(self, noise_std: float = 0.05, num_perturbations: int = 50) -> None:
        """init RelativeInputStability Metric

        Args:
            noise_std (float, optional): standard deviation of noise. Defaults to 0.05.
            num_perturbations (int, optional): number of perturbed samples to generate. Defaults to 50.
        """
        self.noise_std = noise_std
        self.num_perturbations = num_perturbations

    def get_name(self) -> str:
        """get name

        Returns:
            str: name of the metric
        """
        return self.name

    def perturb_input(
        self,
        input: torch.Tensor,
        uq_model: NeuralNetworkBase,
    ) -> torch.Tensor:
        """Perturbs the input by adding Gaussian noise N(0, noise_std^2) and returns num_perturbations samples.

        Args:
            input (torch.Tensor): original input (expected shape: (11,) for Winequality or (1,28, 28) for MNIST)
            uq_model (NeuralNetworkBase): ensemble-based uncertainty quantification model

        Returns:
            torch.Tensor: perturbed inputs with shape (num_perturbations, *input.shape)
        """
        inp = input.clone()
        noise = torch.normal(
            0,
            self.noise_std,
            size=(self.num_perturbations, *inp.shape),
            device=inp.device,
            dtype=inp.dtype,
        )

        perturbed_inputs = inp.unsqueeze(0).expand(self.num_perturbations, *inp.shape) + noise

        perturbed_inputs = self.check_uncertainty_label(
            input=input, perturbed_inputs=perturbed_inputs, uq_model=uq_model
        )
        return perturbed_inputs

    def check_uncertainty_label(
        self,
        input: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        uq_model: NeuralNetworkBase,
        tol: float = 0.05,
    ) -> torch.Tensor:
        """Return only those perturbed inputs whose uncertainty lies within ±tol of the original uncertainty.

        Args:
            input (torch.Tensor): The original input tensor (shape e.g. (D,) or (1,D)).
            perturbed_inputs (torch.Tensor): Tensor of perturbed inputs (shape: (num_perturbations, D)).
            uq_model (NeuralNetworkBase): Uncertainty quantification model.

        Returns:
            torch.Tensor: Subset of `perturbed_inputs` whose uncertainties are within ±tol of the original
            uncertainty. If no perturbed inputs match, an empty tensor with the same trailing dimensions as
            `perturbed_inputs` is returned (i.e., shape (0, D)).
        """

        original_uncertainty = uq_model.epistemic_variance(input)
        perturbed_uncertainties = uq_model.epistemic_variance(perturbed_inputs)

        if torch.is_tensor(original_uncertainty):
            original_uncertainty = original_uncertainty.reshape(-1)[0].to(
                device=perturbed_uncertainties.device, dtype=perturbed_uncertainties.dtype
            )
        else:
            original_uncertainty = torch.tensor(
                original_uncertainty,
                device=perturbed_uncertainties.device,
                dtype=perturbed_uncertainties.dtype,
            )
        mask = (perturbed_uncertainties >= (original_uncertainty - tol)) & (
            perturbed_uncertainties <= (original_uncertainty + tol)
        )
        mask = mask.reshape(-1)

        if mask.numel() == 0 or not mask.any():
            return perturbed_inputs[:0]

        return perturbed_inputs[mask]

    def get_denominator(
        self, original_input: torch.Tensor, perturbed_input: torch.Tensor
    ) -> float:
        """Compute the denominator for the relative input stability metric.

        Args:
            original_input (torch.Tensor): original input
            perturbed_inputs (torch.Tensor): perturbed inputs with shape (num_perturbations, num_features)
        """

        denominator = original_input - perturbed_input
        denominator /= original_input + (original_input == 0) * self.eps

        denominator = np.linalg.norm(denominator.flatten(), 2)
        denominator += (denominator == 0) * self.eps
        return denominator

    def get_numerator(self, original_attr: torch.Tensor, perturbed_attr: torch.Tensor) -> float:
        """Compute the numerator for the relative input stability metric.

        Args:
            original_attr (torch.Tensor): original uncertainty attribution
            perturbed_attr (torch.Tensor): perturbed uncertainty attribution
        """
        temp = (original_attr - perturbed_attr) / (original_attr + (original_attr == 0) * self.eps)
        numerator = float(np.linalg.norm(temp.flatten(), 2))
        return numerator

    def evaluate_uncertainty_attributions(
        self,
        uncertainty_attributions: torch.Tensor,
        X_test: torch.Tensor,
        empirical_xuq_generator: EmpiricalXUQGenerator,
        uq_model: NeuralNetworkBase,
        **kwargs,
    ):
        """Evaluate uncertainty attributions w.r.t. the relative input stability metric

        Args:
            X_test (torch.Tensor): test dataset
            empirical_xuq_generator (EmpiricalXUQGenerator): empirical uncertainty attribution generator
            uq_model (NeuralNetworkBase): uncertainty quantification model

        Returns:
            tuple: metric names and their corresponding values
        """
        RIS_values = []
        nr_perturbations = []
        nr_testsamples = len(X_test)

        assert (
            uq_model.ensemble is not None
        ), "Ensemble models are required for computing uncertainty attributions."

        for i in tqdm(range(nr_testsamples), desc="Computing RIS metric"):
            sample = X_test[i]
            original_attr = uncertainty_attributions[i]
            perturbed_inputs = self.perturb_input(
                input=sample, uq_model=uq_model #TODO ändern
            )
            if perturbed_inputs.shape[0] == 0:
                continue
            if perturbed_inputs.shape[0] < 2:
                warnings.warn("Not enough perturbed inputs for meaningful evaluation.")

            perturbed_attr, _ = empirical_xuq_generator.compute_uncertainty_attr(
                nr_testsamples=perturbed_inputs.shape[0],
                X_test=perturbed_inputs,
                ensemble=uq_model.ensemble,
                pred_test=uq_model.forward_uq(perturbed_inputs)["predicted_classes"]
                if uq_model.model_props.is_classification
                else None,
            )
            max_sensitivity = 0.0
            orig_input_np = sample.detach().cpu().numpy()
            orig_attr_np = original_attr.detach().cpu().numpy()
            pert_attr_np = perturbed_attr.detach().cpu().numpy()

            for j in range(pert_attr_np.shape[0]):
                denominator = self.get_denominator(
                    original_input=orig_input_np,
                    perturbed_input=perturbed_inputs[j].detach().cpu().numpy(),
                )
                nom = self.get_numerator(
                    original_attr=orig_attr_np,
                    perturbed_attr=pert_attr_np[j],
                )
                ratio = nom / denominator
                if ratio > max_sensitivity:
                    max_sensitivity = ratio

            RIS_values.append(max_sensitivity)
            nr_perturbations.append(perturbed_inputs.shape[0])
        attr_dict = {}
        attr_dict["RIS_mean"] = np.array(RIS_values).mean()
        attr_dict["RIS_std"] = np.array(RIS_values).std()

        return attr_dict, (RIS_values, nr_perturbations)
