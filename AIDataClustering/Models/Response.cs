using dataClustering.Models;
using System.Collections.Generic;

namespace AIDataClustering.Models
{
    public class RawResponse
    {
        public RawResponse()
        {
        }

        public List<float> RawClustersCentroids { get; set; }

        public uint ClusterId { get; set; }
    }
}
