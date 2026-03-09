import torch
import time
import os
import numpy as np
import warnings
import random
import pickle
from datetime import datetime

from datasets import mnist
from src.explainer.explainer import (
    InputXGradientExplainer,
    IntegratedGradientsExplainer,
    LRPExplainer,
    GradientSHAPExplainer,
)
from src.utils.utils_data_formatter import save_dict_to_text
from src.evaluation.evaluation_utils import evaluate_uncertainty_attributions
from torch.utils.data import DataLoader, TensorDataset


from src.models.models_dropout import UQMCDropoutCNNClassifier
from src.models.models_dropconnect import UQMCDropconnectCNNClassifier
from src.models.props import MNISTModelProps, ModelTrainingProps
from src.uncertainty_attributions.empirical_xuq import EmpiricalXUQGenerator

from src.evaluation.complexity import Complexity
from src.evaluation.feature_flipping import FeatureFlipping
from src.evaluation.relative_rank_improvement import RelativeRankImprovement
from src.evaluation.uncertainty_conveyance_similarity import UncertaintyConveyanceSimilarity
from src.evaluation.relative_input_stability import RelativeInputStability
from src.evaluation.repeatability import Repeatability

warnings.filterwarnings(
    "ignore",
    message=r"You are providing multiple inputs for Lime.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Objective did not converge.*",
    category=UserWarning,
)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    now = datetime.now()
    print("Starting experiment at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    start_overall = time.time()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    metric_list = [
        Complexity(),
        FeatureFlipping(baseline="gaussian_blur"),
        UncertaintyConveyanceSimilarity(),
        RelativeInputStability(),
        Repeatability(),
        RelativeRankImprovement(
            ood_strategy="patch_inversion", img=True, sample_percentage=25 / 784
        ),
    ]

    analyse_for_samples_with_top_uncertainty = False
    analyse_for_samples_with_low_uncertainty = False

    mc_passes = 50
    uq_strategy = "dropout"  # "dropout" or "dropconnect"
    n_folds = 5

    nr_testsamples = 100

    if uq_strategy not in ["dropout", "dropconnect"]:
        raise AssertionError(f"UQ strategy {uq_strategy} not implemented.")

    folds_dict = mnist.MNISTDataset().serve_dataset_as_folds(
        k_folds=n_folds, val_set_required=True, random_state=seed
    )
    fold_uncertainty_attributions = {}
    fold_uncertainty_metrics = {}

    for key, _ in folds_dict.items():
        split_data = folds_dict[key]
        X_train, X_test, X_val, y_train, y_test, y_val = (
            split_data["X_train"],
            split_data["X_test"],
            split_data["X_val"],
            split_data["y_train"],
            split_data["y_test"],
            split_data["y_val"],
        )

        #########################################################

        model_props = MNISTModelProps(
            ensemble_path="",
            state_dict_path=None,
            preprocessor_path=None,
            target_preprocessor_path=None,
            output_size=10,
            hidden_layer_1=50,
            hidden_layer_2=50,
            hidden_layer_3=None,
            hidden_layer_4=None,
            hidden_layer_5=None,
            hidden_layer_6=None,
            drop_prob=0.2,
            forward_passes=mc_passes,
            last_hidden_layer=50,
            layers_to_perturb=None,
            force_cpu=True,
            is_classification=True,
        )

        training_props = ModelTrainingProps(
            initialisation_strategy=None,
            epochs=10,
            learn_rate=0.001,
            optimizer="Adam",
            loss_function="CEL",
            weight_decay=0.0,
            early_stopping_patience=None,
            lr_scheduler_factor=0.0,
            momentum=0.0,
            batch_size=32,
        )

        ############################################################################################
        explainers = [
            IntegratedGradientsExplainer(),
            InputXGradientExplainer(),
            GradientSHAPExplainer(),
            LRPExplainer(gamma=0.3, cnn=True),
        ]
        metrics_for_all_explainers = {}

        if uq_strategy == "dropout":
            uq_model = UQMCDropoutCNNClassifier(
                model_props=model_props,
                training_props=training_props,
            )
        elif uq_strategy == "dropconnect":
            uq_model = UQMCDropconnectCNNClassifier(
                model_props=model_props,
                training_props=training_props,
            )

        def _ensure_channel(tensor):
            t = tensor.data.float() if hasattr(tensor, "data") else tensor.float()
            if t.dim() == 3:
                t = t.unsqueeze(1)
            return t

        X_train = _ensure_channel(X_train)
        X_test = _ensure_channel(X_test)
        X_val = _ensure_channel(X_val)


        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        train = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test = DataLoader(test_dataset, batch_size=32, shuffle=False)   
        uq_model.fit(
            train,
            val,
        )

        model, ensemble = (
            uq_model.model,
            uq_model.get_ensemble(save_model=False, ensemble_path=None),
        )
        if analyse_for_samples_with_top_uncertainty or analyse_for_samples_with_low_uncertainty:
            test_predictive_uncertainty = uq_model.forward_uq(X_test)["variance"].detach().numpy()
            variance_ordered_list = np.argsort(-test_predictive_uncertainty)

            if analyse_for_samples_with_top_uncertainty:
                X_test, y_test = (
                    X_test[variance_ordered_list[:nr_testsamples]],
                    y_test[variance_ordered_list[:nr_testsamples]],
                )
            elif analyse_for_samples_with_low_uncertainty:
                X_test, y_test = (
                    X_test[variance_ordered_list[-1 * nr_testsamples :]],
                    y_test[variance_ordered_list[-1 * nr_testsamples :]],
                )
        else:
            X_test, y_test = X_test[:nr_testsamples], y_test[:nr_testsamples]
        unpacked_dataset = (X_train, X_test, y_train, y_test)

        pred_tests = uq_model.forward_uq(X_test)["predicted_classes"]

        ############################################################
        uncertainty_attributions_for_explainer = {}
        for explainer in explainers:
            empirical_xuq = EmpiricalXUQGenerator(
                explainer=explainer, uq_strategy=uq_strategy, mc_passes=mc_passes
            )
            print(f"STARTING {explainer.get_name()}")
            start = time.time()

            uncertainty_attributions, _ = empirical_xuq.compute_uncertainty_attr(
                nr_testsamples, X_test, pred_tests, ensemble
            )
            evaluation_metrics = evaluate_uncertainty_attributions(
                uncertainty_attributions=uncertainty_attributions,
                metric_list=metric_list,
                uq_model=uq_model,
                unpacked_dataset=unpacked_dataset,
                explainer=explainer,
                uq_strategy=uq_strategy,
                mc_passes=mc_passes,
                pred_tests=pred_tests,
                base_model=model,
                ensemble=ensemble,
                nr_testsamples=nr_testsamples,
                model_props=model_props,
                X_test=X_test,
                empirical_xuq_generator=empirical_xuq,
            )

            end = time.time()
            print(
                f" time to obtain explanations of uncertainty using {explainer.get_name()} in {key}: {str((end - start) / 60)} minutes"
            )

            uncertainty_attributions = (
                uncertainty_attributions.detach().numpy()
                if isinstance(uncertainty_attributions, torch.Tensor)
                else uncertainty_attributions
            )
            uncertainty_attributions_for_explainer[explainer.get_name()] = uncertainty_attributions
            metrics_for_all_explainers[explainer.get_name()] = evaluation_metrics
            print(f"ENDING {explainer.get_name()}")

        print(metrics_for_all_explainers)

        fold_uncertainty_attributions[key] = uncertainty_attributions_for_explainer
        fold_uncertainty_metrics[key] = metrics_for_all_explainers
        evaluation_metrics_model = uq_model.evaluate_uq(test, calibration_metrics=True)
        fold_uncertainty_metrics[key]["model_evaluation"] = evaluation_metrics_model

    stem_dir = os.path.join(
        "results",
        "mnist",
        date_str,
        time_str,
        "evaluation",
        "cross_validation",
    )
    save_dir = os.path.join(
        stem_dir,
        "{}_samples".format(nr_testsamples),
        uq_strategy,
        "layers_{}".format(model_props.layers_to_perturb),
        "drop_prob" + str(model_props.drop_prob),
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_save_path_pkl = save_dir + "/{}.pkl".format("cross_val_metrics_mnist")

    with open(result_save_path_pkl, "wb") as f:
        pickle.dump(fold_uncertainty_metrics, f)
    save_dict_to_text(fold_uncertainty_metrics, save_dir, "cross_val_metrics_mnist")

    print("Metric Experiment on {} done".format("mnist"))
    end_overall = time.time()
    print(
        f" Overall time to evaluate uncertainty attributions: {str((end_overall - start_overall) / 60)} minutes"
    )
