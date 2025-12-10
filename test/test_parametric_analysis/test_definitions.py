import unittest

from parameteric_evaluation.definitions import (
    PhysicalMetric, OtherParameters, BatteryPowerFlows,
    EnvironmentalMetric, EconomicMetric, LoadMatchingMetric, TimeAggregation
)


class TestDefinitions(unittest.TestCase):
    def test_physical_metric_values(self):
        """Test PhysicalMetric enum values"""
        self.assertEqual(PhysicalMetric.SHARED_ENERGY.value, "Shared energy")
        self.assertEqual(PhysicalMetric.TOTAL_CONSUMPTION.value, "Total consumption")
        self.assertEqual(PhysicalMetric.INVALID.value, "invalid")

    def test_physical_metric_abbreviations(self):
        """Test PhysicalMetric abbreviation mapping"""
        self.assertEqual(PhysicalMetric.SHARED_ENERGY.to_abbrev_str(), "e_sh")
        self.assertEqual(PhysicalMetric.TOTAL_CONSUMPTION.to_abbrev_str(), "c_tot")

    def test_physical_metric_valid(self):
        """Test PhysicalMetric valid method"""
        self.assertTrue(PhysicalMetric.SHARED_ENERGY.valid())
        self.assertFalse(PhysicalMetric.INVALID.valid())

    def test_physical_metric_get_all(self):
        """Test PhysicalMetric get_all method"""
        all_metrics = list(PhysicalMetric.get_all())
        self.assertIn(PhysicalMetric.SHARED_ENERGY, all_metrics)
        self.assertIn(PhysicalMetric.TOTAL_CONSUMPTION, all_metrics)
        self.assertIn(PhysicalMetric.INVALID, all_metrics)

    def test_other_parameters_values(self):
        """Test OtherParameters enum values"""
        self.assertEqual(OtherParameters.INJECTED_ENERGY.value, "Injected energy")
        self.assertEqual(OtherParameters.WITHDRAWN_ENERGY.value, "Withdrawn energy")
        self.assertEqual(OtherParameters.INVALID.value, "invalid")

    def test_other_parameters_abbreviations(self):
        """Test OtherParameters abbreviation mapping"""
        self.assertEqual(OtherParameters.INJECTED_ENERGY.to_abbrev_str(), "e_inj")
        self.assertEqual(OtherParameters.WITHDRAWN_ENERGY.to_abbrev_str(), "e_with")

    def test_battery_power_flows_values(self):
        """Test BatteryPowerFlows enum values"""
        self.assertEqual(BatteryPowerFlows.STORED_ENERGY.value, "Stored energy")
        self.assertEqual(BatteryPowerFlows.POWER_CHARGE.value, "Charging power")

    def test_battery_power_flows_abbreviations(self):
        """Test BatteryPowerFlows abbreviation mapping"""
        self.assertEqual(BatteryPowerFlows.STORED_ENERGY.to_abbrev_str(), "e_stor")
        self.assertEqual(BatteryPowerFlows.POWER_CHARGE.to_abbrev_str(), "p_charge")

    def test_environmental_metric_values(self):
        """Test EnvironmentalMetric enum values"""
        self.assertEqual(EnvironmentalMetric.BASELINE_EMISSIONS.value, "Baseline emissions")
        self.assertEqual(EnvironmentalMetric.TOTAL_EMISSIONS.value, "Total emissions")
        self.assertEqual(EnvironmentalMetric.ESR.value, "Emissions savings ratio")

    def test_environmental_metric_abbreviations(self):
        """Test EnvironmentalMetric abbreviation mapping"""
        self.assertEqual(EnvironmentalMetric.ESR.to_abbrev_str(), "esr")
        self.assertEqual(EnvironmentalMetric.TOTAL_EMISSIONS.to_abbrev_str(), "em_tot")
        self.assertEqual(EnvironmentalMetric.BASELINE_EMISSIONS.to_abbrev_str(), "e_base")

    def test_economic_metric_values(self):
        """Test EconomicMetric enum values"""
        self.assertEqual(EconomicMetric.CAPEX.value, "Capex")
        self.assertEqual(EconomicMetric.OPEX.value, "Opex")

    def test_economic_metric_abbreviations(self):
        """Test EconomicMetric abbreviation mapping"""
        self.assertEqual(EconomicMetric.CAPEX.to_abbrev_str(), "capex")
        self.assertEqual(EconomicMetric.OPEX.to_abbrev_str(), "opex")

    def test_load_matching_metric_values(self):
        """Test LoadMatchingMetric enum values"""
        self.assertEqual(LoadMatchingMetric.SELF_CONSUMPTION.value, "Self consumption")
        self.assertEqual(LoadMatchingMetric.SELF_SUFFICIENCY.value, "Self sufficiency")
        self.assertEqual(LoadMatchingMetric.SELF_PRODUCTION.value, "Self production")
        self.assertEqual(LoadMatchingMetric.GRID_LIABILITY.value, "Grid liability")

    def test_load_matching_metric_abbreviations(self):
        """Test LoadMatchingMetric abbreviation mapping"""
        self.assertEqual(LoadMatchingMetric.SELF_CONSUMPTION.to_abbrev_str(), "sc")
        self.assertEqual(LoadMatchingMetric.SELF_SUFFICIENCY.to_abbrev_str(), "ss")

    def test_time_aggregation_values(self):
        """Test TimeAggregation enum values"""
        self.assertEqual(TimeAggregation.THEORETICAL_LIMIT.value, "15min")
        self.assertEqual(TimeAggregation.HOUR.value, "hour")
        self.assertEqual(TimeAggregation.DAY.value, "dayofyear")
        self.assertEqual(TimeAggregation.MONTH.value, "month")
        self.assertEqual(TimeAggregation.SEASON.value, "season")
        self.assertEqual(TimeAggregation.YEAR.value, "year")

    def test_time_aggregation_abbreviations(self):
        """Test TimeAggregation abbreviation mapping"""
        self.assertEqual(TimeAggregation.HOUR.to_abbrev_str(), "hour")
        self.assertEqual(TimeAggregation.MONTH.to_abbrev_str(), "month")
        self.assertEqual(TimeAggregation.THEORETICAL_LIMIT.to_abbrev_str(), "th_lim")


if __name__ == '__main__':
    unittest.main()

