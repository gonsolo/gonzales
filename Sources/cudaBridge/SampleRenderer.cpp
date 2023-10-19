#include "SampleRenderer.h"

extern "C" char embedded_ptx_code[];

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  TriangleMeshSBTData data;
};


SampleRenderer::SampleRenderer(const Model *model)
  : model(model)
{
  initOptix();
  createContext();
  createModule();
  createRaygenPrograms();
  createMissPrograms();
  createHitgroupPrograms();
  launchParams.traversable = buildAccel();
  createPipeline();
  buildSBT();
  launchParamsBuffer.alloc(sizeof(launchParams));
}

OptixTraversableHandle SampleRenderer::buildAccel()
{
  const int numMeshes = (int)model->meshes.size();
  vertexBuffer.resize(numMeshes);
  normalBuffer.resize(numMeshes);
  texcoordBuffer.resize(numMeshes);
  indexBuffer.resize(numMeshes);
  
  OptixTraversableHandle asHandle { 0 };
  
  // ==================================================================
  // triangle inputs
  // ==================================================================
  std::vector<OptixBuildInput> triangleInput(numMeshes);
  std::vector<CUdeviceptr> d_vertices(numMeshes);
  std::vector<CUdeviceptr> d_indices(numMeshes);
  std::vector<uint32_t> triangleInputFlags(numMeshes);

  for (int meshID=0;meshID<numMeshes;meshID++) {
    // upload the model to the device: the builder
    TriangleMesh &mesh = *model->meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
    indexBuffer[meshID].alloc_and_upload(mesh.index);
    if (!mesh.normal.empty())
      normalBuffer[meshID].alloc_and_upload(mesh.normal);
    if (!mesh.texcoord.empty())
      texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

    triangleInput[meshID] = {};
    triangleInput[meshID].type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    d_indices[meshID]  = indexBuffer[meshID].d_pointer();
    
    triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertex.size();
    triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
  
    triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
    triangleInput[meshID].triangleArray.numIndexTriplets    = (int)mesh.index.size();
    triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
  
    triangleInputFlags[meshID] = 0 ;
  
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords               = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
  }
  // ==================================================================
  // BLAS setup
  // ==================================================================
  
  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
    | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
    ;
  accelOptions.motionOptions.numKeys  = 1;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
  
  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage
              (optixContext,
               &accelOptions,
               triangleInput.data(),
               (int)numMeshes,  // num_build_inputs
               &blasBufferSizes
               ));
  
  // ==================================================================
  // prepare compaction
  // ==================================================================
  
  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));
  
  OptixAccelEmitDesc emitDesc;
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();
  
  // ==================================================================
  // execute build (main stage)
  // ==================================================================
  
  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
  
  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
    
  OPTIX_CHECK(optixAccelBuild(optixContext,
                              /* stream */0,
                              &accelOptions,
                              triangleInput.data(),
                              (int)numMeshes,
                              tempBuffer.d_pointer(),
                              tempBuffer.sizeInBytes,
                              
                              outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes,
                              
                              &asHandle,
                              
                              &emitDesc,1
                              ));
  CUDA_SYNC_CHECK();
  
  // ==================================================================
  // perform compaction
  // ==================================================================
  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize,1);
  
  asBuffer.alloc(compactedSize);
  OPTIX_CHECK(optixAccelCompact(optixContext,
                                /*stream:*/0,
                                asHandle,
                                asBuffer.d_pointer(),
                                asBuffer.sizeInBytes,
                                &asHandle));
  CUDA_SYNC_CHECK();
  
  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================
  outputBuffer.free(); // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();
  
  return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void SampleRenderer::initOptix()
{
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("no CUDA capable devices found!");
  std::cout << "Found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK( optixInit() );
  std::cout << GDT_TERMINAL_GREEN
            << "Successfully initialized optix!"
            << GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
  //fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

/*! creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
void SampleRenderer::createContext()
{
  // for this sample, do everything on one device
  const int deviceID = 0;
  CUDA_CHECK(SetDevice(deviceID));
  CUDA_CHECK(StreamCreate(&stream));
    
  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout << "Running on device: " << deviceProps.name << std::endl;
    
  CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
  if( cuRes != CUDA_SUCCESS ) 
    fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
    
  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback
              (optixContext,context_log_cb,nullptr,4));
}



/*! creates the module that contains all the programs we are going
  to use. in this simple example, we use a single module from a
  single .cu file, using a single embedded ptx string */
void SampleRenderer::createModule()
{
  moduleCompileOptions.maxRegisterCount  = 50;
  moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur     = false;
  pipelineCompileOptions.numPayloadValues   = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
    
  pipelineLinkOptions.maxTraceDepth          = 2;
    
  const std::string ptxCode = embedded_ptx_code;
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixModuleCreate(optixContext,
                                       &moduleCompileOptions,
                                       &pipelineCompileOptions,
                                       ptxCode.c_str(),
                                       ptxCode.size(),
                                       log,&sizeof_log,
                                       &module
                                       ));
  //if (sizeof_log > 1) PRINT(log);
}
  


/*! does all setup for the raygen program(s) we are going to use */
void SampleRenderer::createRaygenPrograms()
{
  // we do a single ray gen program in this example:
  raygenPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module            = module;           
  pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &raygenPGs[0]
                                      ));
  //if (sizeof_log > 1) PRINT(log);
}
  
/*! does all setup for the miss program(s) we are going to use */
void SampleRenderer::createMissPrograms()
{
  // we do a single ray gen program in this example:
  missPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module            = module;           
  pgDesc.miss.entryFunctionName = "__miss__radiance";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &missPGs[0]
                                      ));
  //if (sizeof_log > 1) PRINT(log);
}
  
/*! does all setup for the hitgroup program(s) we are going to use */
void SampleRenderer::createHitgroupPrograms()
{
  // for this simple example, we set up a single hit group
  hitgroupPGs.resize(1);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH            = module;
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.moduleAH            = module;
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &hitgroupPGs[0]
                                      ));
  //if (sizeof_log > 1) PRINT(log);
}
  

/*! assembles the full pipeline of all programs */
void SampleRenderer::createPipeline()
{
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto pg : missPGs)
    programGroups.push_back(pg);
  for (auto pg : hitgroupPGs)
    programGroups.push_back(pg);
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixPipelineCreate(optixContext,
                                  &pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups.data(),
                                  (int)programGroups.size(),
                                  log,&sizeof_log,
                                  &pipeline
                                  ));
  //if (sizeof_log > 1) PRINT(log);

  OPTIX_CHECK(optixPipelineSetStackSize
              (/* [in] The pipeline to configure the stack size for */
               pipeline, 
               /* [in] The direct stack size requirement for direct
                  callables invoked from IS or AH. */
               2*1024,
               /* [in] The direct stack size requirement for direct
                  callables invoked from RG, MS, or CH.  */                 
               2*1024,
               /* [in] The continuation stack requirement. */
               2*1024,
               /* [in] The maximum depth of a traversable graph
                  passed to trace. */
               1));
  //if (sizeof_log > 1) PRINT(log);
}


/*! constructs the shader binding table */
void SampleRenderer::buildSBT()
{
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0;i<raygenPGs.size();i++) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (int i=0;i<missPGs.size();i++) {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  missRecordsBuffer.alloc_and_upload(missRecords);
  sbt.missRecordBase          = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount         = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  int numObjects = (int)model->meshes.size();
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int meshID=0;meshID<numObjects;meshID++) {
    auto mesh = model->meshes[meshID];
    
    HitgroupRecord rec;
    // all meshes use the same code, so all same hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
    rec.data.color   = mesh->diffuse;
    if (mesh->diffuseTextureID >= 0) {
      rec.data.hasTexture = true;
      rec.data.texture    = textureObjects[mesh->diffuseTextureID];
    } else {
      rec.data.hasTexture = false;
    }
    rec.data.index    = (vec3i*)indexBuffer[meshID].d_pointer();
    rec.data.vertex   = (vec3f*)vertexBuffer[meshID].d_pointer();
    rec.data.normal   = (vec3f*)normalBuffer[meshID].d_pointer();
    rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
    hitgroupRecords.push_back(rec);
  }
  hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}



/*! render one frame */
void SampleRenderer::render()
{
  // sanity check: make sure we launch only after first resize is
  // already done:
  if (launchParams.frame.size.x == 0) return;
    
  launchParamsBuffer.upload(&launchParams,1);
    
  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline,stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.d_pointer(),
                          launchParamsBuffer.sizeInBytes,
                          &sbt,
                          /*! dimensions of the launch: */
                          launchParams.frame.size.x,
                          launchParams.frame.size.y,
                          1
                          ));
  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void SampleRenderer::setCamera(const Camera &camera)
{
  lastSetCamera = camera;
  launchParams.camera.position  = camera.from;
  launchParams.camera.rayDirection = camera.rayDirection;
  launchParams.camera.tHit = camera.tHit;
}

/*! resize frame buffer to given resolution */
void SampleRenderer::resize(const vec2i &newSize)
{
  // if window minimized
  if (newSize.x == 0 | newSize.y == 0) return;
  
  // resize our cuda frame buffer
  colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
  outVertexBuffer.resize(newSize.x*newSize.y*sizeof(vec3f));
  outNormalBuffer.resize(newSize.x*newSize.y*sizeof(vec3f));
  intersectedBuffer.resize(sizeof(int));
  primIDBuffer.resize(sizeof(int));

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.frame.size  = newSize;
  launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
  launchParams.frame.outVertexBuffer = (vec3f*)outVertexBuffer.d_pointer();
  launchParams.frame.outNormalBuffer = (vec3f*)outNormalBuffer.d_pointer();
  launchParams.frame.intersected = (int*)intersectedBuffer.d_pointer();
  launchParams.frame.primID = (int*)primIDBuffer.d_pointer();

  // and re-set the camera, since aspect may have changed
  setCamera(lastSetCamera);
}

/*! download the rendered color buffer */
void SampleRenderer::downloadPixels(
  uint32_t h_pixels[],
  vec3f h_vertices[],
  vec3f h_normals[],
  int h_intersected[],
  int h_primID[])
{
  colorBuffer.download(h_pixels, launchParams.frame.size.x*launchParams.frame.size.y);
  outVertexBuffer.download(h_vertices, launchParams.frame.size.x*launchParams.frame.size.y);
  outNormalBuffer.download(h_normals, launchParams.frame.size.x*launchParams.frame.size.y);
  intersectedBuffer.download(h_intersected, 1);
  primIDBuffer.download(h_primID, 1);
}
