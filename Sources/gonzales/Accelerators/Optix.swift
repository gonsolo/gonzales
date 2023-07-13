import cuda

func cudaCheck(_ cudaError: cudaError_t) {
        if cudaError != cudaSuccess {
                print("Cuda error: \(cudaError)")
        }
}

func printCudaDevices() {
        var numDevices: Int32 = 0
        var cudaError: cudaError_t
        cudaError = cudaGetDeviceCount(&numDevices)
        cudaCheck(cudaError)
        print("Cuda device count: \(numDevices)")

        var cudaDevice: Int32 = 0
        cudaError = cudaGetDevice(&cudaDevice)
        cudaCheck(cudaError)
        print("Cuda device used: \(cudaDevice)")

        var cudaDeviceProperties: cudaDeviceProp = cudaDeviceProp()
        cudaError = cudaGetDeviceProperties_v2(&cudaDeviceProperties, cudaDevice)
        cudaCheck(cudaError)

        let name = withUnsafePointer(to: cudaDeviceProperties.name) {
                $0.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout.size(ofValue: $0)) {
                        String(cString: $0)
                }
        }
        print(name)
}