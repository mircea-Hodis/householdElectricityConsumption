using AIDataClustering.Services;
using dataClustering.Services;
using Microsoft.AspNetCore.Mvc;

namespace AIDataClustering.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class PredictionController : ControllerBase
    {
        private KmeansService _service;
        private KMeansAnuallyServiceRaw _anuallyServiceRaw;
        public PredictionController(KmeansService kmeansService, KMeansAnuallyServiceRaw rawService)
        {
            _service = kmeansService;
            _anuallyServiceRaw = rawService;
        }

        [HttpGet]
        public void Get(int numberOfClusters)
        {
            _service.GenerateCluseters(numberOfClusters);
        }

        [HttpGet]
        [Route("GetRaw")]
        public JsonResult GetRaw(int numberOfClusters, int consumption)
        {
            var result = _anuallyServiceRaw.GenerateCluseters(numberOfClusters, consumption);

            return new JsonResult(result);
        }
    }
}
