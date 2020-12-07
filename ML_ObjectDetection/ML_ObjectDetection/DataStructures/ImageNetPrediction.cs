using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_ObjectDetection.DataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
