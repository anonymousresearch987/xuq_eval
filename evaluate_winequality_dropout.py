import torch
import time
import os
import pickle
import warnings
import random
import numpy as np
from datetime import datetime

from datasets import winequality
from src.explainer.explainer import (
    ShapleyValueSamplingExplainer,
    InputXGradientExplainer,
    IntegratedGradientsExplainer,
    LRPExplainer,
    LimeExplainer,
)
from torch.utils.data import DataLoader, TensorDataset

from src.utils.utils_data_formatter import save_dict_to_text
from src.evaluation.evaluation_utils import evaluate_uncertainty_attributions

from src.models.models_dropconnect import UQMCDropconnectRegressor
from src.models.models_dropout import UQMCDropoutRegressor
from src.models.props import WineQualityModelProps, ModelTrainingProps
from src.uncertainty_attributions.empirical_xuq import EmpiricalXUQGenerator

from src.evaluation.complexity import Complexity
from src.evaluation.feature_flipping import FeatureFlipping
from src.evaluation.relative_rank_improvement import RelativeRankImprovement
from src.evaluation.repeatability import Repeatability
from src.evaluation.uncertainty_conveyance_similarity import UncertaintyConveyanceSimilarity
from src.evaluation.relative_input_stability import RelativeInputStability


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
    start_overall = time.time()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    metric_list = [
        Complexity(),
        FeatureFlipping(baseline="kde"),
        Repeatability(),
        UncertaintyConveyanceSimilarity(),
        RelativeInputStability(),
        RelativeRankImprovement(ood_strategy="sigma_based", img=False, sample_percentage=1),
    ]
    analyse_for_samples_with_top_uncertainty = False
    analyse_for_samples_with_low_uncertainty = False

    mc_passes = 50
    uq_strategy = "dropout"  # "dropout" or "dropconnect"
    n_folds = 5

    nr_testsamples = 5

    if uq_strategy not in ["dropout", "dropconnect"]:
        raise AssertionError(f"UQ strategy {uq_strategy} not implemented.")

    folds_dict = winequality.WineQualityDataset().serve_dataset_as_folds(
        n_folds, val_set_required=True
    )
    fold_uncertainty_attributions = {}
    fold_uncertainty_metrics = {}

    for key, values in folds_dict.items():
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

        model_props = WineQualityModelProps(
            ensemble_path="",
            state_dict_path=None,
            preprocessor_path=None,
            target_preprocessor_path=None,
            input_size=11,
            output_size=1,
            hidden_layer_1=50,
            hidden_layer_2=50,
            hidden_layer_3=None,
            hidden_layer_4=None,
            hidden_layer_5=None,
            hidden_layer_6=None,
            drop_prob=0.1,
            forward_passes=50,
            last_hidden_layer=50,
            layers_to_perturb=None,
            force_cpu=True,
            is_classification=False,
        )

        training_props = ModelTrainingProps(
            initialisation_strategy=None,
            epochs=5,
            learn_rate=0.001,
            optimizer="Adam",
            loss_function="MSE",
            weight_decay=0.0,
            early_stopping_patience=None,
            lr_scheduler_factor=0.0,
            momentum=0.0,
            batch_size=32,
        )

        ############################################################################################
        explainers = [
            LRPExplainer(gamma=0.3),
            IntegratedGradientsExplainer(),
            ShapleyValueSamplingExplainer(X_train=X_train, nsamples=25),
            InputXGradientExplainer(),
            LimeExplainer(num_samples=25),
        ]
        metrics_for_all_explainers = {}

        all_uncertainty_attributions = {}
        if uq_strategy == "dropout":
            uq_model = UQMCDropoutRegressor(model_props=model_props, training_props=training_props)
            print("Using the Dropout Model")
        elif uq_strategy == "dropconnect":
            uq_model = UQMCDropconnectRegressor(
                model_props=model_props, training_props=training_props
            )
            print("Using the DropConnect Model")


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
        if len(X_test.shape) >= 3:
            X_test = X_test.squeeze()
        if len(y_test.shape) >= 3:
            y_test = y_test.squeeze()
        unpacked_dataset = (X_train, X_test, y_train, y_test)
        ############################################################
        for explainer in explainers:
            empirical_xuq = EmpiricalXUQGenerator(explainer, uq_strategy)
            print(f"STARTING {explainer.get_name()}")
            start = time.time()
            uncertainty_attributions_for_explainer = {}
            metrics_for_explainer = {}

            uncertainty_attributions, mean_explanations = empirical_xuq.compute_uncertainty_attr(
                nr_testsamples, X_test, pred_test=None, ensemble=ensemble
            )
            evaluation_metrics = evaluate_uncertainty_attributions(
                uncertainty_attributions=uncertainty_attributions,
                metric_list=metric_list,
                uq_model=uq_model,
                unpacked_dataset=unpacked_dataset,
                explainer=explainer,
                uq_strategy=uq_strategy,
                mc_passes=mc_passes,
                pred_tests=None,  # prediction set to None, if the model is a regression model it doesn't need a target oriented explanation
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
        "winequality",
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

    result_save_path_pkl = save_dir + "/{}.pkl".format("cross_val_metrics_winequality")

    with open(result_save_path_pkl, "wb") as f:
        pickle.dump(fold_uncertainty_metrics, f)
    save_dict_to_text(fold_uncertainty_metrics, save_dir, "cross_val_metrics_winequality")

    print("Metric Experiment on {} done".format("winequality"))
    end_overall = time.time()
    print(
        f" Overall time to evaluate uncertainty attributions: {str((end_overall - start_overall) / 60)} minutes"
    )
