from opbench.config import load_config


def test_example_config_contains_no_credentials():
    config = load_config("configs/evaluation.example.yaml")
    assert config.agent.api_key == ""
    assert config.scorer.api_key == ""
    assert config.agent_ports["Caroline"] == 8096

