// Generated evidence-linked diabetes knowledge graph
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Type 1'}) MERGE (s)-[:HAS_TYPE {evidence:'D1'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Type 2'}) MERGE (s)-[:HAS_TYPE {evidence:'D1'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Gestational diabetes'}) MERGE (s)-[:HAS_TYPE {evidence:'D1'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'High blood glucose'}) MERGE (s)-[:CHARACTERIZED_BY {evidence:'D1'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Thirst'}) MERGE (s)-[:HAS_SYMPTOM {evidence:'D2'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Frequent urination'}) MERGE (s)-[:HAS_SYMPTOM {evidence:'D2'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Fatigue'}) MERGE (s)-[:HAS_SYMPTOM {evidence:'D2'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Unexplained weight loss'}) MERGE (s)-[:HAS_SYMPTOM {evidence:'D2'}]->(o);
MERGE (s:Concept {name:'Diabetes'}) MERGE (o:Concept {name:'Blurred vision'}) MERGE (s)-[:HAS_SYMPTOM {evidence:'D2'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Family history'}) MERGE (s)-[:HAS_RISK_FACTOR {evidence:'D3'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Overweight'}) MERGE (s)-[:HAS_RISK_FACTOR {evidence:'D3'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Low physical activity'}) MERGE (s)-[:HAS_RISK_FACTOR {evidence:'D3'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Older age'}) MERGE (s)-[:HAS_RISK_FACTOR {evidence:'D3'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Prediabetes'}) MERGE (s)-[:HAS_RISK_FACTOR {evidence:'D6'}]->(o);
MERGE (s:Concept {name:'Type 2'}) MERGE (o:Concept {name:'Physical activity'}) MERGE (s)-[:CAN_BE_DELAYED_BY {evidence:'D4'}]->(o);
