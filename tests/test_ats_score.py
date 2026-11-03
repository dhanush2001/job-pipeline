"""Tests for the ATS keyword-match scorer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ats_score import score_resume


def test_match_rate_and_missing():
    jd = "We need Python, Kubernetes, SQL and machine learning experience."
    resume = {"skills": [{"category": "x", "entries": "Python, SQL, pandas"}],
              "summary": "Machine learning engineer."}
    r = score_resume(resume, jd)
    # posting skills: python, kubernetes, sql, machine learning -> resume has 3/4
    assert r["score"] == 75, r
    assert "kubernetes" in r["missing"] and "kubernetes" not in r["matched"]
    assert "python" in r["matched"]
    print("test_match_rate_and_missing: PASS")


def test_word_boundary_no_false_positive():
    # "java" must not be counted just because "javascript" is present
    r = score_resume({"summary": "I use JavaScript."}, "Need Java and JavaScript.")
    assert r["score"] == 50, r
    assert "java" in r["missing"] and "javascript" in r["matched"]
    print("test_word_boundary_no_false_positive: PASS")


def test_none_when_posting_has_no_skills():
    r = score_resume({"summary": "hello"}, "We value teamwork and clear communication.")
    assert r["score"] is None
    print("test_none_when_posting_has_no_skills: PASS")


def test_synonyms_count_as_the_same_skill():
    """'OOP' in the posting is satisfied by 'Object-Oriented Programming' on the
    resume (and vice versa), and the two spellings count as ONE skill."""
    resume = {"education": "Coursework: Object-Oriented Programming, Data Structures"}
    r = score_resume(resume, "Strong OOP and data structures fundamentals.")
    assert "object-oriented" in r["matched"], r
    assert "oop" not in r["missing"] and "object-oriented" not in r["missing"], r
    assert r["score"] == 100, r
    # Both spellings in the posting -> still one skill, not two.
    r2 = score_resume(resume, "OOP / object-oriented design and data structures.")
    assert r2["matched"].count("object-oriented") == 1 and r2["score"] == 100, r2
    print("test_synonyms_count_as_the_same_skill: PASS")


def test_spring_season_is_not_the_framework():
    """'spring 2028' is a semester, not the Spring framework."""
    r = score_resume({"summary": "Python developer."},
                     "Python. Graduating fall 2027 or spring 2028. Spring semester start.")
    assert "spring" not in r["matched"] and "spring" not in r["missing"], r
    # The real framework is still detected.
    r2 = score_resume({"summary": "Built services in Spring."}, "Java with Spring and Spring Boot.")
    assert "spring" in r2["matched"], r2
    print("test_spring_season_is_not_the_framework: PASS")


def test_either_or_list_is_one_requirement():
    """'Go, Ruby, or Java' with Java on the resume is satisfied -- it used to
    score 33% and list Go and Ruby as missing."""
    r = score_resume({"skills": "Java"}, "Experience with Go, Ruby, or Java.")
    assert r["score"] == 100 and r["missing"] == [] and r["matched"] == ["java"], r
    assert r["alternatives"] == [["go", "java", "ruby"]]
    # Unmet group -> ONE missing entry, not three.
    r2 = score_resume({"skills": "Python"}, "Experience with Go, Ruby, or Java.")
    assert r2["score"] == 0 and r2["missing"] == ["go or java or ruby"], r2
    print("test_either_or_list_is_one_requirement: PASS")


def test_and_lists_and_slashes_stay_separate():
    r = score_resume({"skills": "Python"}, "Python, SQL, and Spark required.")
    assert r["score"] == 33 and r["missing"] == ["spark", "sql"], r
    r2 = score_resume({"skills": "Python"}, "Daily work in Python/SQL.")
    assert r2["missing"] == ["sql"], r2          # slash usually means both
    print("test_and_lists_and_slashes_stay_separate: PASS")


def test_claimable_missing_expands_groups():
    from src.ats_score import claimable_missing
    assert claimable_missing(["go or java or ruby", "sql", "kubernetes"], {"java", "sql"}) == ["java", "sql"]
    print("test_claimable_missing_expands_groups: PASS")


if __name__ == "__main__":
    test_either_or_list_is_one_requirement()
    test_and_lists_and_slashes_stay_separate()
    test_claimable_missing_expands_groups()
    test_match_rate_and_missing()
    test_word_boundary_no_false_positive()
    test_none_when_posting_has_no_skills()
    test_synonyms_count_as_the_same_skill()
    test_spring_season_is_not_the_framework()
    print("\nAll ATS-score tests passed.")


def test_languages_the_posting_says_can_be_learned_are_alternatives():
    # Stripe New Grad: the stack list is context, not five separate requirements.
    jd = ("&lt;li&gt;We work mostly in Java, Ruby, JavaScript, Scala, and Go. We believe new "
          "programming languages can be learned.&lt;/li&gt;&lt;li&gt;Experience with SQL&lt;/li&gt;")
    r = score_resume({"skills": "Java, SQL"}, jd)
    assert r["score"] == 100 and r["missing"] == []
    # Without the can-be-learned line, an "and" list is still each required.
    r = score_resume({"skills": "Java"}, "We use Java, Ruby, and Go.")
    assert sorted(r["missing"]) == ["go", "ruby"]
