#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"
#include "include/vec.h"

struct OptixCamera {
        VectorVec3f from;
        VectorVec3f rayDirection;
        VectorFloat tHit;
};

class OptixRenderer {
      public:
        OptixRenderer(const Model *model);

        void render();

        void resize(const vec2i &newSize);

        void downloadPixels(uint32_t h_pixels[], vec3f h_vertices[], vec3f h_normals[], int h_intersected[],
                            int h_primID[], float h_tMax[]);

        void setCamera(const OptixCamera &camera);

      protected:
        void initOptix();

        void createContext();

        void createModule();

        void createRaygenPrograms();

        void createMissPrograms();

        void createHitgroupPrograms();

        void createPipeline();

        void buildSBT();

        OptixTraversableHandle buildAccel();

        void createTextures();

      protected:
        CUcontext cudaContext;
        CUstream stream;
        cudaDeviceProp deviceProps;

        OptixDeviceContext optixContext;

        OptixPipeline pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions pipelineLinkOptions = {};

        OptixModule module;
        OptixModuleCompileOptions moduleCompileOptions = {};

        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;

        CUDABuffer colorBuffer;
        CUDABuffer outVertexBuffer;
        CUDABuffer outNormalBuffer;
        CUDABuffer intersectedBuffer;
        CUDABuffer primIDBuffer;
        CUDABuffer tMaxBuffer;

        CUDABuffer cameraPositionBuffer;
        CUDABuffer cameraDirectionBuffer;
        CUDABuffer cameraTHitBuffer;

        const Model *model;

        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> normalBuffer;
        std::vector<CUDABuffer> texcoordBuffer;
        std::vector<CUDABuffer> indexBuffer;

        CUDABuffer asBuffer;

        std::vector<cudaArray_t> textureArrays;
        std::vector<cudaTextureObject_t> textureObjects;
};
