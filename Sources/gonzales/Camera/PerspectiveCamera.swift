@MainActor
var numberCameraRays = 0

final class PerspectiveCamera: Camera, Transformable {

        @MainActor
        init(
                cameraToWorld: Transform,
                screenWindow: Bounds2f,
                fov: FloatX,
                focalDistance: FloatX,
                lensRadius: FloatX,
                film: Film
        ) async throws {
                self.objectToWorld = cameraToWorld
                self.fieldOfView = fov
                self.focalDistance = focalDistance
                self.lensRadius = lensRadius
                self.film = film
                let perspective = try Transform.makePerspective(fov: fov, near: 0.01, far: 1000.0)
                let resolution = await film.getResolution()
                cameraTransform = try CameraTransform(
                        cameraToScreen: perspective,
                        screenWindow: screenWindow,
                        resolution: resolution
                )
        }

        private func depthOfField(ray: Ray, cameraSample: CameraSample) -> Ray {
                guard lensRadius > 0 else {
                        return ray
                }
                let lens = lensRadius * concentricSampleDisk(u: cameraSample.lens)
                let ft = focalDistance / ray.direction.z
                let pFocus = ray.getPointFor(parameter: ft)
                let origin = Point(x: lens.x, y: lens.y, z: 0)
                let direction: Vector = normalized(pFocus - ray.origin)
                return Ray(origin: origin, direction: direction, cameraSample: cameraSample)
        }

        func generateRay(cameraSample: CameraSample) -> Ray {
                let rasterPoint = Point(x: cameraSample.film.0, y: cameraSample.film.1, z: 0)
                let cameraPoint = cameraTransform.rasterToCamera * rasterPoint
                let pinholeRay = Ray(
                        origin: origin,
                        direction: normalized(Vector(point: cameraPoint)),
                        cameraSample: cameraSample
                )
                let ray = depthOfField(ray: pinholeRay, cameraSample: cameraSample)
                //numberCameraRays += 1
                return objectToWorld * ray
        }

        @MainActor
        static func statistics() {
                print("  Camera rays traced:\t\t\t\t\t\t\t\(numberCameraRays)")
        }

        let fieldOfView: FloatX
        let focalDistance: FloatX
        let lensRadius: FloatX
        let cameraTransform: CameraTransform
        let objectToWorld: Transform
        let film: Film
}
