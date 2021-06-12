using dataClustering.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace dataClustering.Services
{
    public class KmeansService
    {
        private MLContext _context;

        public KmeansService()
        {
            _context = new MLContext();
        }

        private IDataView ReadAnually()
        {
            string currentDirectory = Directory.GetCurrentDirectory();
            var path = string.Concat(currentDirectory, "\\Services\\Data\\4_ResElecAnnual_DataJam.csv");
            var trainingData = _context.Data.LoadFromTextFile<YearlyConsumption>(
            path: path,
            hasHeader: true,
            separatorChar: ',');

            return trainingData;
        }

        public void GenerateCluseters(int numberOfClusters)
        {
            var data = ReadAnually();
            
            var pipeline = _context.Transforms
                .Concatenate(
              "Features",
              "HouseholdId",
              "ElectricityUsage")
              .Append(_context.Clustering.Trainers.KMeans("Features", numberOfClusters: numberOfClusters));

            var transformer = pipeline.Fit(data);
            var modelPath = string.Concat("models/", Guid.NewGuid());

            var lastTransformer = transformer.LastTransformer;

            KMeansModelParameters kparams = (KMeansModelParameters)
                lastTransformer.GetType().GetProperty("Model").GetValue(lastTransformer);
            
            VBuffer<float>[] centroids = default;

            kparams.GetClusterCentroids(ref centroids, out var k);

            _context.Model.Save(transformer, data.Schema, modelPath);

            var firstCentroid = centroids[0];

            var values = firstCentroid.GetValues();

            var predictor = _context.Model.CreatePredictionEngine<YearlyConsumption, ClusterPrediction>(transformer);

            var prediction = predictor.Predict(new YearlyConsumption
            {
                HouseholdId = 1,
                ElectricityUsage = 19703
            });

            var centroits = _context.Model;
        }
    }
}
