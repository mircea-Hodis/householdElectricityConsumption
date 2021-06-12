using AIDataClustering.Models;
using dataClustering.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace AIDataClustering.Services
{
    public class KMeansAnuallyServiceRaw
    {
        private MLContext _context;
        private RawResponse _response;

        public KMeansAnuallyServiceRaw()
        {
            _context = new MLContext();
            _response = new RawResponse();
        }

        public RawResponse GenerateCluseters(int numberOfClusters, int consumptionValue)
        {
            var data = ReadAnually();

            TrainTestData dataSplit = _context.Data.TrainTestSplit(data, 0.4);
           
            var pipeline = _context.Transforms
             .Concatenate("ElectricityUsage", "AnnualElec_kWh")
             .Append(_context.Clustering.Trainers.KMeans("ElectricityUsage", numberOfClusters: numberOfClusters));

            Train(dataSplit.TrainSet, numberOfClusters, pipeline);
         
            var transformer = pipeline.Fit(dataSplit.TestSet);
            var modelPath = string.Concat("models/", Guid.NewGuid());

            _context.Model.Save(transformer, data.Schema, modelPath);

            var predictor = _context.Model.CreatePredictionEngine<RawYearlyConsumption, ClusterPrediction>(transformer);

            _response.ClusterId = predictor.Predict(new RawYearlyConsumption
            {
                AnnualElec_kWh = consumptionValue
            }).PredictedClusterId;

            GetClustersCentroids(transformer);

            return _response;
        }
        
        private void Train(IDataView trainData, int numberOfClusters, EstimatorChain<ClusteringPredictionTransformer<KMeansModelParameters>> pipeline)
        {
            pipeline.Fit(trainData);

            ITransformer dataPrepTransformer = pipeline.Fit(trainData);

            dataPrepTransformer.Transform(trainData);
        }

        private IDataView ReadAnually()
        {
            string currentDirectory = Directory.GetCurrentDirectory();
            var path = string.Concat(currentDirectory, "\\Services\\Data\\4_ResElecAnnual_DataJam.csv");
            var data = _context.Data.LoadFromTextFile<RawYearlyConsumption>(
                path: path,
                hasHeader: true,
                separatorChar: ',');

            return data;
        }

        private void GetClustersCentroids(
            TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>> transformer)
        {
            var lastTransformer = transformer.LastTransformer;

            KMeansModelParameters kparams = (KMeansModelParameters)
            lastTransformer.GetType().GetProperty("Model").GetValue(lastTransformer);

            VBuffer<float>[] centroids = default;

            kparams.GetClusterCentroids(ref centroids, out var k);

            GetClusterRawCentroidValues(centroids);
        }

        private void GetClusterRawCentroidValues(VBuffer<float>[] clusters)
        {
            var values = new List<float>();

            foreach(var centroid in clusters)
            {
                var items = centroid.Items(true);
                var consumptionItem = items.First();
                values.Add(consumptionItem.Value);
            }

            _response.RawClustersCentroids = values;
        }
    }
}
