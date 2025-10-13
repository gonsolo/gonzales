final class Tile: Sendable {

        init(integrator: VolumePathIntegrator, bounds: Bounds2i) {
                self.integrator = integrator
                self.bounds = bounds
        }

        func render(
                reporter: ProgressReporter,
                sampler: RandomSampler,
                camera: any Camera,
                scene: Scene,
                lightSampler: LightSampler,
                state: ImmutableState
        ) throws -> [Sample] {
                var samples = [Sample]()
                for pixel in bounds {
                        for _ in 0..<sampler.samplesPerPixel {
                                let cameraSample = sampler.getCameraSample(pixel: pixel)
                                let ray = camera.generateRay(cameraSample: cameraSample)
                                var tHit: FloatX = Float.infinity
                                let radianceAlbedoNormal = try integrator.getRadianceAndAlbedo(
                                        from: ray,
                                        tHits: &tHit,
                                        with: sampler,
                                        lightSampler: lightSampler,
                                        state: state)

                                let radiance = radianceAlbedoNormal.0
                                let albedo = radianceAlbedoNormal.1
                                let normal = radianceAlbedoNormal.2
                                let rayWeight: FloatX = 1.0
                                let sample = Sample(
                                        light: radiance,
                                        albedo: albedo,
                                        normal: normal,
                                        weight: rayWeight,
                                        location: Point2f(x: cameraSample.film.0, y: cameraSample.film.1))
                                samples.append(sample)

                        }
                }
                return samples
        }

        let integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
