import Vapor

private struct AvailabilityRequest: Content {
    let acceptingStorage: Bool

    enum CodingKeys: String, CodingKey {
        case acceptingStorage = "accepting_storage"
    }
}

private struct AvailabilityResponse: Content {
    let accepted: Bool
}

@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
func registerAvailabilityRoute(
    _ app: RoutesBuilder,
    registrationClient: MothershipRegistrationClient
) {
    app.post("v1", "availability") { req async throws -> AvailabilityResponse in
        let body = try req.content.decode(AvailabilityRequest.self)
        await registrationClient.sendAvailabilityUpdate(acceptingStorage: body.acceptingStorage)
        return AvailabilityResponse(accepted: true)
    }
}
