from services.sustainability import (
    calculate_sustainability,
)


def assert_score_range(result: dict) -> None:
    assert 0.0 <= result["sustainability_score"] <= 1.0

    assert 0.0 <= (
        result["nitrogen"]["nitrogen_score"]
    ) <= 1.0

    assert 0.0 <= (
        result["soil"]["soil_health_score"]
    ) <= 1.0

    assert 0.0 <= (
        result["ghg"]["ghg_score"]
    ) <= 1.0


def test_good_sustainability_case() -> None:
    result = calculate_sustainability(
        crop="wheat",
        farming_strategy="reduced",
        machine_use="low",
        fertilizer_type="can",
        fertilizer_quantity_kg_ha=400.0,
        soil_ph=6.5,
        soil_soc_g_kg=22.0,
        soil_clay_pct=25.0,
        expected_yield_t_ha=5.0,
        yield_source="test",
    )

    assert_score_range(result)

    assert (
        result["sustainability_score_100"]
        >= 75
    )


def test_medium_sustainability_case() -> None:
    result = calculate_sustainability(
        crop="maize",
        farming_strategy="conventional",
        machine_use="medium",
        fertilizer_type="urea",
        fertilizer_quantity_kg_ha=300.0,
        soil_ph=5.7,
        soil_soc_g_kg=14.0,
        soil_clay_pct=40.0,
        expected_yield_t_ha=8.0,
        yield_source="test",
    )

    assert_score_range(result)

    assert (
        45
        <= result[
            "sustainability_score_100"
        ]
        <= 75
    )


def test_poor_sustainability_case() -> None:
    result = calculate_sustainability(
        crop="maize",
        farming_strategy="reduced",
        machine_use="high",
        fertilizer_type="urea",
        fertilizer_quantity_kg_ha=400.0,
        soil_ph=5.0,
        soil_soc_g_kg=8.0,
        soil_clay_pct=55.0,
        expected_yield_t_ha=7.0,
        yield_source="test",
    )

    assert_score_range(result)

    assert (
        result["sustainability_score_100"]
        <= 50
    )


def test_case_ordering() -> None:
    good = calculate_sustainability(
        crop="wheat",
        farming_strategy="reduced",
        machine_use="low",
        fertilizer_type="can",
        fertilizer_quantity_kg_ha=400.0,
        soil_ph=6.5,
        soil_soc_g_kg=22.0,
        soil_clay_pct=25.0,
        expected_yield_t_ha=5.0,
        yield_source="test",
    )

    medium = calculate_sustainability(
        crop="maize",
        farming_strategy="conventional",
        machine_use="medium",
        fertilizer_type="urea",
        fertilizer_quantity_kg_ha=300.0,
        soil_ph=5.7,
        soil_soc_g_kg=14.0,
        soil_clay_pct=40.0,
        expected_yield_t_ha=8.0,
        yield_source="test",
    )

    poor = calculate_sustainability(
        crop="maize",
        farming_strategy="reduced",
        machine_use="high",
        fertilizer_type="urea",
        fertilizer_quantity_kg_ha=400.0,
        soil_ph=5.0,
        soil_soc_g_kg=8.0,
        soil_clay_pct=55.0,
        expected_yield_t_ha=7.0,
        yield_source="test",
    )

    assert (
        good["sustainability_score"]
        > medium["sustainability_score"]
        > poor["sustainability_score"]
    )


def test_unsupported_strategy() -> None:
    try:
        calculate_sustainability(
            crop="wheat",
            farming_strategy="precision",
            machine_use="low",
            fertilizer_type="can",
            fertilizer_quantity_kg_ha=300.0,
            soil_ph=6.5,
            soil_soc_g_kg=20.0,
            soil_clay_pct=25.0,
            expected_yield_t_ha=5.0,
            yield_source="test",
        )

        assert False

    except ValueError as error:
        assert (
            "Unsupported farming strategy"
            in str(error)
        )


def test_print_example_results() -> None:
    cases = {
        "GOOD": {
            "crop": "wheat",
            "farming_strategy": "reduced",
            "machine_use": "low",
            "fertilizer_type": "can",
            "fertilizer_quantity_kg_ha": 400.0,
            "soil_ph": 6.5,
            "soil_soc_g_kg": 22.0,
            "soil_clay_pct": 25.0,
            "expected_yield_t_ha": 5.0,
            "yield_source": "test",
        },
        "MEDIUM": {
            "crop": "maize",
            "farming_strategy": "conventional",
            "machine_use": "medium",
            "fertilizer_type": "urea",
            "fertilizer_quantity_kg_ha": 300.0,
            "soil_ph": 5.7,
            "soil_soc_g_kg": 14.0,
            "soil_clay_pct": 40.0,
            "expected_yield_t_ha": 8.0,
            "yield_source": "test",
        },
        "POOR": {
            "crop": "maize",
            "farming_strategy": "reduced",
            "machine_use": "high",
            "fertilizer_type": "urea",
            "fertilizer_quantity_kg_ha": 400.0,
            "soil_ph": 5.0,
            "soil_soc_g_kg": 8.0,
            "soil_clay_pct": 55.0,
            "expected_yield_t_ha": 7.0,
            "yield_source": "test",
        },
    }

    for name, values in cases.items():
        result = calculate_sustainability(
            **values
        )

        print()
        print("=" * 50)
        print(name)
        print("=" * 50)

        print(
            "Sustainability:",
            result[
                "sustainability_score_100"
            ],
        )

        print(
            "Nitrogen:",
            round(
                result["nitrogen"][
                    "nitrogen_score"
                ] * 100
            ),
        )

        print(
            "Soil:",
            round(
                result["soil"][
                    "soil_health_score"
                ] * 100
            ),
        )

        print(
            "GHG:",
            round(
                result["ghg"][
                    "ghg_score"
                ] * 100
            ),
        )

        print(
            "Applied N:",
            round(
                result["nitrogen"][
                    "applied_nitrogen_kg_ha"
                ],
                2,
            ),
        )

        print(
            "N balance:",
            round(
                result["nitrogen"][
                    "nitrogen_balance_kg_ha"
                ],
                2,
            ),
        )

def test_high_conventional_machinery_has_score_floor() -> None:
    result = calculate_sustainability(
        crop="maize",
        farming_strategy="conventional",
        machine_use="high",
        fertilizer_type="urea",
        fertilizer_quantity_kg_ha=250.0,
        soil_ph=6.5,
        soil_soc_g_kg=20.0,
        soil_clay_pct=25.0,
        expected_yield_t_ha=7.0,
        yield_source="test",
    )

    assert (
        result["ghg"]["machinery_emission_score"]
        == 0.10
    )

    assert result["ghg"]["ghg_score"] > 0.0
    assert result["sustainability_score"] > 0.0