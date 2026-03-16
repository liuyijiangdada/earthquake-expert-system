# 地震知识图谱Schema设计

# 实体类型
ENTITY_TYPES = {
    "Earthquake": {
        "properties": [
            "id",
            "name",
            "time",
            "magnitude",
            "depth",
            "location",
            "latitude",
            "longitude",
            "intensity",
            "description"
        ]
    },
    "Region": {
        "properties": [
            "id",
            "name",
            "province",
            "city",
            "district",
            "population",
            "area"
        ]
    },
    "Magnitude": {
        "properties": [
            "id",
            "value",
            "level",
            "description"
        ]
    },
    "Depth": {
        "properties": [
            "id",
            "value",
            "unit",
            "category"
        ]
    },
    "Time": {
        "properties": [
            "id",
            "timestamp",
            "date",
            "time",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second"
        ]
    }
}

# 关系类型
RELATIONSHIP_TYPES = {
    "OCCURRED_IN": {
        "source": "Earthquake",
        "target": "Region",
        "properties": ["distance"]
    },
    "HAS_MAGNITUDE": {
        "source": "Earthquake",
        "target": "Magnitude",
        "properties": []
    },
    "HAS_DEPTH": {
        "source": "Earthquake",
        "target": "Depth",
        "properties": []
    },
    "OCCURRED_AT": {
        "source": "Earthquake",
        "target": "Time",
        "properties": []
    },
    "LOCATED_IN": {
        "source": "Region",
        "target": "Region",
        "properties": ["level"]
    }
}

# 知识图谱查询模板
QUERY_TEMPLATES = {
    "get_earthquakes_by_region": "MATCH (e:Earthquake)-[r:OCCURRED_IN]->(rgn:Region) WHERE rgn.name = $region RETURN e",
    "get_earthquakes_by_magnitude": "MATCH (e:Earthquake)-[r:HAS_MAGNITUDE]->(m:Magnitude) WHERE m.value >= $min_magnitude RETURN e",
    "get_earthquakes_by_time_range": "MATCH (e:Earthquake)-[r:OCCURRED_AT]->(t:Time) WHERE t.timestamp >= $start_time AND t.timestamp <= $end_time RETURN e",
    "get_related_earthquakes": "MATCH (e1:Earthquake)-[r1:OCCURRED_IN]->(rgn:Region)<-[r2:OCCURRED_IN]-(e2:Earthquake) WHERE e1.id = $earthquake_id RETURN e2",
    "get_earthquake_details": "MATCH (e:Earthquake)-[r1:OCCURRED_IN]->(rgn:Region), (e)-[r2:HAS_MAGNITUDE]->(m:Magnitude), (e)-[r3:HAS_DEPTH]->(d:Depth), (e)-[r4:OCCURRED_AT]->(t:Time) WHERE e.id = $earthquake_id RETURN e, rgn, m, d, t"
}