def test_package_imports():
    import pbrl_emoa
    from pbrl_emoa.algorithms.pbrl_emoa import DDPGEMOA
    from pbrl_emoa.preference.machine_dm import MachineDM
    from pbrl_emoa.preference.rank_svm import RankSVM
    from pbrl_emoa.preference.value_functions import linear_value_function

    assert pbrl_emoa.__version__ == "0.1.0"
    assert DDPGEMOA is not None
    assert MachineDM is not None
    assert RankSVM is not None
    assert linear_value_function is not None
