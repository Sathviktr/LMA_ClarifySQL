"""Small made-up records; never connects to a university's real systems."""

import sqlite3
from pathlib import Path
from clarifysql.episodes.schema import Episode

GLOSSARY = [{"term": "high marks", "definition": "marks above 80", "aliases": ["high mark"]}]


def create_database(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER REFERENCES departments(id), age INTEGER);
        CREATE TABLE marks (student_id INTEGER REFERENCES students(id), mark REAL);
        CREATE TABLE placements (student_id INTEGER REFERENCES students(id), placed INTEGER);
        INSERT INTO departments VALUES (1,'Computing'),(2,'Mathematics'),(3,'Physics');
        INSERT INTO students VALUES (1,'Asha',1,21),(2,'Ben',1,23),(3,'Chen',2,20),(4,'Dana',2,24),(5,'Eli',3,22),(6,'Fara',3,25);
        INSERT INTO marks VALUES (1,90),(2,84),(3,92),(4,88),(5,78),(6,86);
        INSERT INTO placements VALUES (1,1),(2,1),(3,0),(4,1),(5,0),(6,1);
        """)
    return path


def examples() -> list[Episode]:
    clear = Episode("demo-clear", "demo", "university_demo", "clear", "List the names of all departments.",
                    gold_sql="SELECT name FROM departments", source_question="List the names of all departments.")
    clear.reference = [{"action": "execute", "request": clear.request}]
    definition = Episode("demo-definition", "demo", "university_demo", "definition", "List students with high marks.",
        glossary=GLOSSARY, gold_sql="SELECT s.name FROM students s JOIN marks m ON m.student_id=s.id WHERE m.mark > 80",
        source_question="List students with marks above 80.")
    definition.reference = [{"action": "inspect", "tool": "search_definitions", "args": {"query": "high marks"}},
                            {"action": "execute", "request": definition.source_question}]
    ranking = {"kind": "ranking", "term": "best department", "value": "average mark", "hidden": "highest average mark",
               "answer": "Rank departments by average mark.", "meanings": ["average mark", "placement rate"]}
    ambiguous = Episode("demo-ambiguous", "demo", "university_demo", "ambiguous", "Which is the best department?",
        slots=[ranking], source_question="Which department has the highest average mark?",
        gold_sql="SELECT d.name FROM departments d JOIN students s ON s.department_id=d.id JOIN marks m ON m.student_id=s.id GROUP BY d.id ORDER BY avg(m.mark) DESC LIMIT 1")
    ambiguous.reference = [{"action": "clarify", "question": "Should the best department be ranked by marks or placements?",
                             "slots": [{"name": "ranking", "term": ranking["term"], "meanings": ranking["meanings"]}]},
                            {"action": "execute", "request": ambiguous.source_question}]
    two = Episode("demo-two-slot", "demo", "university_demo", "two_slot", "List older students with strong marks.",
        source_question="List students older than 22 with marks above 85.",
        gold_sql="SELECT s.name FROM students s JOIN marks m ON m.student_id=s.id WHERE s.age > 22 AND m.mark > 85",
        slots=[{"kind": "threshold", "term": "older students", "value": "> 22", "hidden": "older than 22",
                "answer": "Use age above 22.", "meanings": ["age above a missing cutoff"]},
               {"kind": "threshold", "term": "strong marks", "value": "> 85", "hidden": "above 85",
                "answer": "Use marks above 85.", "meanings": ["mark above a missing cutoff"]}])
    two.reference = [{"action": "clarify", "question": f"What counts as {s['term']}?",
                      "slots": [{"name": "threshold", "term": s["term"], "meanings": s["meanings"]}]} for s in two.slots]
    two.reference.append({"action": "execute", "request": two.source_question})
    missing = Episode("demo-unanswerable", "demo", "university_demo", "unanswerable", "List students' email addresses.",
                      missing={"attribute": "email address"})
    missing.reference = [{"action": "inspect", "tool": "describe_table", "args": {"table": "students"}},
                         {"action": "abstain", "missing": "The database has no student email addresses."}]
    return [clear, definition, ambiguous, two, missing]
