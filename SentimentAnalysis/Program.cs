using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Legacy;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath  = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed:0);
            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text,1)
                }
            });

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            Predict(mlContext, model);

            SaveModelAsFile(mlContext, model);

            Console.ReadLine();
        }

        private static void Predict(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = model.MakePredictionFunction<SentimentData, SentimentPrediction>(mlContext);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This is a very rude movie"
            };

            var resultprediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
            //throw new NotImplementedException();
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            //throw new NotImplementedException();
            IDataView dataView = _textLoader.Read(_testDataPath);
            Console.WriteLine("Evaluating Model with test data");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static ITransformer Train(MLContext mlContext, string trainDataPath)
        {
            IDataView dataView = _textLoader.Read(trainDataPath);
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves:50, numTrees:50, minDatapointsInLeafs: 20));

            Console.WriteLine("Creating and training the model");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Completed Training");

            return model;
            //throw new NotImplementedException();
        }
    }
}
