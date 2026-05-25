//
//  Database.USer.swift
//  database-server
//
//  Created by Ritesh Pakala Rao on 12/15/25.
//

import Vapor

extension Database {
    struct User: Content, Codable {
        var groups: [Database.Group]
    }
}
