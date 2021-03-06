struct CameraTransform {

        public init(cameraToScreen: Transform,
                    screenWindow: Bounds2f,
                    resolution: Point2I) throws {
                self.cameraToScreen = cameraToScreen
                self.screenToRaster = try Transform.makeScale(x: FloatX(resolution.x),
                                                      y: FloatX(resolution.y),
                                                      z: 1) *
                                      Transform.makeScale(x: 1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                                                      y: 1 / (screenWindow.pMin.y - screenWindow.pMax.y),
                                                      z: 1) *
                                      Transform.makeTranslation(from: Vector(x: -screenWindow.pMin.x,
                                                                           y: -screenWindow.pMax.y, z: 0))
                self.rasterToScreen = screenToRaster.inverse
                self.rasterToCamera = try cameraToScreen.inverse * rasterToScreen
        }

        public var cameraToScreen: Transform
        public var screenToRaster: Transform
        public var rasterToCamera: Transform
        public var rasterToScreen: Transform
}

