// Generated evidence-linked diabetes knowledge graph
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Weight management'}) MERGE (s)-[:CAN_BE_DELAYED_BY {evidence:'D4'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Healthy eating'}) MERGE (s)-[:CAN_BE_DELAYED_BY {evidence:'D4'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Heart'}) MERGE (s)-[:MAY_AFFECT {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Kidneys'}) MERGE (s)-[:MAY_AFFECT {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Eyes'}) MERGE (s)-[:MAY_AFFECT {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Feet'}) MERGE (s)-[:MAY_AFFECT {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Blood vessels'}) MERGE (s)-[:MAY_AFFECT {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'High blood glucose'}) MERGE (o:Concept {name:'Glucose management'}) MERGE (s)-[:MANAGED_BY {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Glucose management'}) MERGE (o:Concept {name:'Heart'}) MERGE (s)-[:HELPS_REDUCE_RISK_FOR {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Glucose management'}) MERGE (o:Concept {name:'Kidneys'}) MERGE (s)-[:HELPS_REDUCE_RISK_FOR {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Glucose management'}) MERGE (o:Concept {name:'Eyes'}) MERGE (s)-[:HELPS_REDUCE_RISK_FOR {evidence:'D5'}]->(o);
MERGE (s:Concept {name:'Physical activity'}) MERGE (o:Concept {name:'Weight management'}) MERGE (s)-[:SUPPORTS {evidence:'D4'}]->(o);
MERGE (s:Concept {name:'Healthy eating'}) MERGE (o:Concept {name:'Weight management'}) MERGE (s)-[:SUPPORTS {evidence:'D4'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'No clear symptoms'}) MERGE (s)-[:MAY_HAVE {evidence:'D6'}]->(o);
MERGE (s:Concept {name:'Prediabetes'}) MERGE (o:Concept {name:'Type 2'}) MERGE (s)-[:INCREASES_RISK_OF {evidence:'D6'}]->(o);
MERGE (s:Concept {name:'Family history'}) MERGE (o:Concept {name:'Type 2'}) MERGE (s)-[:INCREASES_RISK_OF {evidence:'D6'}]->(o);
