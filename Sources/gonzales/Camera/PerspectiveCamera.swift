final class PerspectiveCamera: Camera, Transformable {
        
        init(cameraToWorld: Transform, screenWindow: Bounds2f, fov: FloatX, focalDistance: FloatX, lensRadius: FloatX, film: Film) throws {
                self.objectToWorld = cameraToWorld
                self.fieldOfView = fov
                self.focalDistance = focalDistance
                self.lensRadius = lensRadius
                self.film = film
                let perspective = try Transform.makePerspective(fov: fov, near: 0.01, far: 1000.0)
                cameraTransform = try CameraTransform(cameraToScreen: perspective,
                                                      screenWindow: screenWindow,
                                                      resolution: film.image.fullResolution)
        }
        
        func generateRay(sample: CameraSample) -> Ray {

                func depthOfField(ray: inout Ray) {
                        let lens = lensRadius * concentricSampleDisk(u: sample.lens)
                        let ft = focalDistance / ray.direction.z
                        let pFocus = ray.getPointFor(parameter: ft)
                        ray.origin = Point(x: lens.x, y: lens.y, z: 0)
                        ray.direction = normalized(pFocus - ray.origin)
                }

                //print("Camera position: ", objectToWorld * origin)
                let rasterPoint = Point(x: sample.film.x, y: sample.film.y, z: 0)
                let cameraPoint = cameraTransform.rasterToCamera * rasterPoint
                var ray = Ray(origin: origin, direction: normalized(Vector(point: cameraPoint)))
                if lensRadius > 0 { depthOfField(ray: &ray) }
                numberCameraRays += 1
                return objectToWorld * ray
        }

        func statistics() {
                print("  Camera rays generated:\t\t\t\t\t\t\(numberCameraRays)")
        }

        let fieldOfView: FloatX
        let focalDistance: FloatX
        let lensRadius: FloatX
        var numberCameraRays = 0
        let cameraTransform: CameraTransform
        let objectToWorld: Transform
        let film: Film
}

