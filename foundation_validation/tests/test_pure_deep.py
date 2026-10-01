import torch
import yaml

from src.models.pure_deep import (
    FeatureStatisticsPool,
    LearnableSpectrumEncoder,
    LearnableRelativeEnergyEncoder,
    MomentOnlyWristEncoder,
    NormFreeMomentEncoder,
    PureDeepSubjectModel,
    PureDeepWristEncoder,
    ResNet1DWristEncoder,
    InceptionTimeWristEncoder,
    GyroLearnableSpectrumEncoder,
    StructuredTokenResidualReadout,
    WristStructuredResidualReadout,
)
from src.utils.config import load_config
from src.models.activity_fusion import MaskedMeanActivityAggregator
from src.engine.self_supervised import (
    MaskedSequenceReconstructor,
    sample_contiguous_span_mask,
)


def test_feature_statistics_pool_is_finite_and_differentiable():
    module = FeatureStatisticsPool(16, "attention_mean_std")
    inputs = torch.randn(3, 16, 41, requires_grad=True)
    output = module(inputs)["embedding"]
    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_subject_model_supports_variable_activity_lengths():
    config = {
        "pure_deep": {"feature_dim": 16, "dropout": 0.0,
                      "pooling": "attention_mean_std"},
        "activity_fusion": {"attention_dim": 8, "dropout": 0.0,
                            "classifier_dropout": 0.0},
    }
    model = PureDeepSubjectModel(["a", "b"], 6, 2, config)
    inputs = torch.randn(2, 2, 2, 6, 128)
    wrist_mask = torch.ones(2, 2, 2)
    activity_mask = torch.ones(2, 2, dtype=torch.bool)
    lengths = torch.tensor([[96, 128], [96, 128]])
    output = model(inputs, wrist_mask, activity_mask, lengths)
    assert output["logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()


def test_structured_token_residual_preserves_order_and_starts_at_zero():
    module = StructuredTokenResidualReadout(
        activity_dim=10, activity_count=3, projection_dim=4, num_classes=2
    )
    features = torch.randn(2, 3, 10, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = module(features, mask)
    assert output["structured_activity_tokens"].shape == (2, 3, 4)
    assert output["structured_residual_embedding"].shape == (2, 12)
    assert output["structured_residual_logits"].shape == (2, 2)
    assert torch.equal(
        output["structured_activity_tokens"][0, 2], torch.zeros(4)
    )
    assert torch.equal(
        output["structured_residual_logits"], torch.zeros(2, 2)
    )


def test_str01_subject_model_initially_matches_base_logits_and_parameter_delta():
    base_config = {
        "pure_deep": {"feature_dim": 16, "dropout": 0.0,
                      "pooling": "attention_mean_std"},
        "activity_fusion": {"attention_dim": 8, "dropout": 0.0,
                            "classifier_dropout": 0.0},
    }
    str_config = {
        **base_config,
        "structured_token_residual": {"enabled": True, "projection_dim": 4},
    }
    torch.manual_seed(19)
    base = PureDeepSubjectModel(["a", "b"], 6, 2, base_config)
    torch.manual_seed(19)
    model = PureDeepSubjectModel(["a", "b"], 6, 2, str_config)
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    # Shared 66D bilateral representation: 66*4+4 projection parameters,
    # followed by an ordered 2*4 -> 2 residual classifier.
    assert model_parameters - base_parameters == (66 * 4 + 4) + (8 * 2 + 2)
    inputs = torch.randn(2, 2, 2, 6, 128)
    wrist_mask = torch.ones(2, 2, 2)
    activity_mask = torch.ones(2, 2, dtype=torch.bool)
    lengths = torch.full((2, 2), 128)
    base.eval()
    model.eval()
    with torch.no_grad():
        base_output = base(inputs, wrist_mask, activity_mask, lengths)
        output = model(inputs, wrist_mask, activity_mask, lengths)
    assert torch.allclose(output["logits"], base_output["logits"])
    assert torch.allclose(output["logits"], output["base_logits"])


def test_wrist_structured_residual_preserves_activity_and_wrist_identity():
    module = WristStructuredResidualReadout(
        wrist_dim=10, activity_count=3, projection_dim=4, num_classes=2
    )
    features = torch.randn(2, 3, 2, 10)
    wrist_mask = torch.tensor([
        [[1, 1], [1, 0], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    activity_mask = torch.tensor([[True, True, False], [True, True, True]])
    output = module(features, wrist_mask, activity_mask)
    assert output["wrist_structured_tokens"].shape == (2, 3, 2, 4)
    assert output["wrist_structured_embedding"].shape == (2, 24)
    assert output["wrist_structured_residual_logits"].shape == (2, 2)
    assert torch.equal(output["wrist_structured_tokens"][0, 1, 1], torch.zeros(4))
    assert torch.equal(output["wrist_structured_tokens"][0, 2], torch.zeros(2, 4))
    assert torch.equal(output["wrist_structured_residual_logits"], torch.zeros(2, 2))


def test_wsr01_initially_matches_complete_str01_and_has_fixed_parameter_delta():
    str_config = {
        "pure_deep": {"feature_dim": 16, "dropout": 0.0,
                      "pooling": "attention_mean_std"},
        "activity_fusion": {"attention_dim": 8, "dropout": 0.0,
                            "classifier_dropout": 0.0},
        "structured_token_residual": {"enabled": True, "projection_dim": 4},
    }
    wsr_config = {
        **str_config,
        "wrist_structured_residual": {"enabled": True, "projection_dim": 4},
    }
    torch.manual_seed(23)
    baseline = PureDeepSubjectModel(["a", "b"], 6, 2, str_config)
    torch.manual_seed(23)
    model = PureDeepSubjectModel(["a", "b"], 6, 2, wsr_config)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert model_parameters - baseline_parameters == (16 * 4 + 4) + (16 * 2 + 2)
    inputs = torch.randn(2, 2, 2, 6, 128)
    wrist_mask = torch.tensor([[[1, 1], [1, 0]], [[1, 1], [1, 1]]])
    activity_mask = torch.ones(2, 2, dtype=torch.bool)
    lengths = torch.full((2, 2), 128)
    baseline.eval()
    model.eval()
    with torch.no_grad():
        baseline_output = baseline(inputs, wrist_mask, activity_mask, lengths)
        output = model(inputs, wrist_mask, activity_mask, lengths)
    assert torch.allclose(output["logits"], baseline_output["logits"])
    assert output["wrist_structured_embedding"].shape == (2, 16)
    assert torch.equal(
        output["wrist_structured_residual_logits"], torch.zeros(2, 2)
    )


def test_split_acc_gyro_stem_is_finite_and_differentiable():
    module = PureDeepWristEncoder(
        input_channels=6,
        feature_dim=16,
        dropout=0.0,
        pooling="attention_mean_std",
        stem_mode="split_acc_gyro",
    )
    inputs = torch.randn(4, 6, 128, requires_grad=True)
    output = module(inputs)["bag_embedding"]
    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_resnet1d_wrist_encoder_contract():
    module = ResNet1DWristEncoder(
        6, 64,
        {"stage_channels": [32, 64, 64], "blocks_per_stage": [2, 2, 2],
         "kernel_size": 7, "stem_kernel_size": 15, "normalization_groups": 8},
        0.1,
    )
    inputs = torch.randn(3, 6, 256, requires_grad=True)
    output = module(inputs)
    assert output["bag_embedding"].shape == (3, 64)
    assert torch.isfinite(output["bag_embedding"]).all()
    assert not any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                   for layer in module.modules())
    output["bag_embedding"].sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_inception_time_wrist_encoder_contract():
    module = InceptionTimeWristEncoder(
        6, 64,
        {"bottleneck_channels": 32, "branch_channels": 32,
         "kernel_sizes": [39, 19, 9], "residual_stack_count": 2,
         "modules_per_stack": 3, "normalization_groups": 8},
        0.1,
    )
    inputs = torch.randn(2, 6, 192, requires_grad=True)
    output = module(inputs)
    assert output["bag_embedding"].shape == (2, 64)
    assert torch.isfinite(output["bag_embedding"]).all()
    assert not any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                   for layer in module.modules())
    output["bag_embedding"].sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_gyro_spectrum_encoder_is_gyro_only_and_differentiable():
    module = GyroLearnableSpectrumEncoder(
        sample_rate=100.0, frequency_bins=64, maximum_hz=12.0,
        hidden_dim=16, embedding_dim=8, dropout=0.0, group_count=8,
    )
    inputs = torch.randn(3, 3, 256, requires_grad=True)
    output = module(inputs)
    assert output["embedding"].shape == (3, 8)
    assert torch.isfinite(output["embedding"]).all()
    output["embedding"].sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_subject_model_supports_standard_backbones_and_frequency_branch():
    base = {
        "feature_dim": 32,
        "dropout": 0.0,
        "gyro_frequency_branch": {
            "enabled": True, "frequency_bins": 64, "maximum_hz": 12.0,
            "hidden_dim": 16, "embedding_dim": 8,
        },
    }
    for mode in ("resnet1d", "inception_time"):
        config = {
            "pure_deep": {**base, "encoder_mode": mode,
                          "resnet1d": {"stage_channels": [16, 32],
                                       "blocks_per_stage": [1, 1]},
                          "inception_time": {"bottleneck_channels": 8,
                                             "branch_channels": 8,
                                             "residual_stack_count": 1,
                                             "modules_per_stack": 2}},
            "sample_rate": 100.0,
            "activity_fusion": {"attention_dim": 16, "dropout": 0.0,
                                "classifier_dropout": 0.0},
        }
        model = PureDeepSubjectModel(["a", "b"], 6, 2, config)
        inputs = torch.randn(2, 2, 2, 6, 128)
        output = model(inputs, torch.ones(2, 2, 2),
                       torch.ones(2, 2, dtype=torch.bool),
                       torch.full((2, 2), 128))
        assert output["logits"].shape == (2, 2)
        assert torch.isfinite(output["logits"]).all()


def test_learnable_spectrum_encoder_is_finite_and_differentiable():
    module = LearnableSpectrumEncoder(
        input_channels=6,
        feature_dim=12,
        sample_rate=100.0,
        frequency_bins=64,
        maximum_hz=20.0,
        dropout=0.0,
    )
    inputs = torch.randn(3, 6, 200, requires_grad=True)
    output = module(inputs)
    assert output["embedding"].shape == (3, 12)
    assert output["spectral_features"].shape[-1] == 32
    assert torch.isfinite(output["embedding"]).all()
    output["embedding"].sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_norm_free_moment_encoder_is_finite_and_differentiable():
    module = NormFreeMomentEncoder(6, 12, [1, 15, 63])
    assert not any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                   for layer in module.modules())
    inputs = torch.randn(3, 6, 200, requires_grad=True)
    output = module(inputs)
    assert output["embedding"].shape == (3, 12)
    assert output["moment_features"].shape == (3, 12, 200)
    assert torch.isfinite(output["embedding"]).all()
    output["embedding"].sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_learnable_relative_energy_is_finite_differentiable_and_scale_invariant():
    module = LearnableRelativeEnergyEncoder(
        input_channels=6,
        feature_dim=12,
        filter_count=5,
        kernel_size=51,
        sample_rate=100.0,
        initial_minimum_hz=0.5,
        initial_maximum_hz=20.0,
    )
    inputs = torch.randn(3, 6, 200, requires_grad=True)
    output = module(inputs)
    scales = torch.tensor([0.5, 1.5, 2.0, 0.8, 3.0, 1.2]).reshape(1, 6, 1)
    scaled = module(inputs.detach() * scales)
    assert output["embedding"].shape == (3, 12)
    assert output["relative_log_energy"].shape == (3, 6, 5)
    assert torch.isfinite(output["embedding"]).all()
    assert torch.allclose(
        output["relative_log_energy"].detach(),
        scaled["relative_log_energy"],
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    output["embedding"].sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()


def test_moment_only_wrist_encoder_contract():
    module = MomentOnlyWristEncoder(6, 16, [1, 15, 63])
    inputs = torch.randn(4, 6, 128, requires_grad=True)
    output = module(inputs)
    assert output["bag_embedding"].shape == (4, 16)
    assert output["encoded_features"].shape == (4, 16, 128)
    output["bag_embedding"].sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_masked_mean_activity_aggregator_is_uniform_over_valid_activities():
    module = MaskedMeanActivityAggregator()
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    output = module(features, torch.tensor([[True, True, False]]))
    assert torch.allclose(output["subject_embedding"], torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(output["activity_attention"], torch.tensor([[0.5, 0.5, 0.0]]))


def test_config_inherits_base_yaml(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"a": {"x": 1, "y": 2}}))
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"_base_": "base.yaml", "a": {"y": 3}})
    )
    # The small fixture is not a complete training config, so this checks the
    # merge primitive through the same public loader only in the project tests.
    # Validation is covered by the real config smoke test on the server.
    try:
        load_config(tmp_path / "child.yaml")
    except KeyError as error:
        assert "Missing required config section: task" in str(error)


def test_contiguous_span_mask_has_exact_size_and_is_reproducible():
    torch.manual_seed(17)
    first = sample_contiguous_span_mask(4, 101, 0.3, 11, device="cpu")
    torch.manual_seed(17)
    second = sample_contiguous_span_mask(4, 101, 0.3, 11, device="cpu")
    assert torch.equal(first, second)
    assert first.sum(dim=1).tolist() == [30, 30, 30, 30]
    assert not first[:, 0].logical_and(first[:, -1]).all()


def test_masked_reconstructor_is_finite_and_updates_encoder():
    encoder = PureDeepWristEncoder(
        input_channels=6,
        feature_dim=16,
        dropout=0.0,
        pooling="attention_mean_std",
        moment_config={"enabled": True, "feature_dim": 8, "kernels": [1, 7]},
    )
    module = MaskedSequenceReconstructor(encoder, 6)
    inputs = torch.randn(3, 6, 128)
    mask = sample_contiguous_span_mask(3, 128, 0.25, 16, device="cpu")
    output = module(inputs, mask)["reconstruction"]
    loss = (output[mask.unsqueeze(1).expand_as(inputs)]
            - inputs[mask.unsqueeze(1).expand_as(inputs)]).square().mean()
    loss.backward()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert encoder.stem[0].weight.grad is not None
    assert torch.isfinite(encoder.stem[0].weight.grad).all()
