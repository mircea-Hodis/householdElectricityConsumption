using Microsoft.ML.Data;
using System;

namespace dataClustering.Models
{
    public class MonthlyConsumption
    {
        public int HouseholdId { get; set; }
        public int Year { get; set; }
        public int Month { get; set; }
        public int ElectricityUsage { get; set; }
    }

    public class YearlyConsumption
    {
        [LoadColumn(0)]
        public Single HouseholdId { get; set; }
        [LoadColumn(2)]
        public Single ElectricityUsage { get; set; }
    }

    public class ClusterPrediction 
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] ElectricityUsage { get; set; }
    }

    public class RawClusterPrediction
    {
        [ColumnName("Score")]
        public float[] ElectricityUsage { get; set; }
    }

    public class RawYearlyConsumption
    {
        [LoadColumn(2)]
        public Single AnnualElec_kWh { get; set; }
    }
}
