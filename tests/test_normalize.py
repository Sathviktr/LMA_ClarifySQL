from clarifysql.data.normalize import normalize_sql


def test_whitespace_and_keyword_case_are_ignored():
    a = "SELECT  name FROM singer   WHERE age > 20;"
    b = "select name from singer where age>20"
    assert normalize_sql(a) == normalize_sql(b)


def test_literal_case_is_kept():
    assert normalize_sql("SELECT * FROM t WHERE x = 'Ward'") != normalize_sql("select * from t where x = 'ward'")


def test_double_quoted_literals_become_single_quoted():
    assert normalize_sql('SELECT * FROM t WHERE c = "Asia"') == normalize_sql("SELECT * FROM t WHERE c = 'Asia'")


def test_quotes_inside_literals_survive():
    assert normalize_sql('SELECT * FROM t WHERE n = "O\'Neil"') == "select * from t where n='O''Neil'"


def test_spaces_inside_literals_are_kept():
    assert "'New  York'" in normalize_sql("SELECT * FROM t WHERE city = 'New  York'")
