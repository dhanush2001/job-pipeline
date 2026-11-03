"""A master-resume project and the repo it was built from are one project:
linked, then judged and written from both sources together."""
from src.github_projects import link_repos
from src.tailor_resume import project_candidates

MR = {"projects": [
    {"name": "MedVQA: Medical Data Platform", "technologies": "Python",
     "bullets": ["Built a FastAPI serving layer"]},
    {"name": "Inference Server", "github": "https://github.com/me/tiny-serve",
     "bullets": ["Built a KV cache"]},
    {"name": "Benchmark", "bullets": ["Ran 15,000 model runs"]},
]}
REPOS = [
    {"name": "MedVQA", "description": "Biomedical serving platform", "readme_excerpt": "controller/worker"},
    {"name": "tiny-serve", "description": "LLM inference server", "readme_excerpt": "paged attention"},
    {"name": "Starlight", "description": "A game"},
]


def test_links_by_name_prefix_and_explicit_field():
    linked = {r["name"]: r.get("same_project_as") for r in link_repos(MR, REPOS)}
    assert linked["MedVQA"] == "MedVQA: Medical Data Platform"
    assert linked["tiny-serve"] == "Inference Server"      # via `github:` field
    assert linked["Starlight"] is None


def test_candidates_combine_both_sources_once():
    cands = {c["name"]: c for c in project_candidates(MR, link_repos(MR, REPOS))}
    # Linked repos are folded in, not offered a second time.
    assert set(cands) == {"MedVQA: Medical Data Platform", "Inference Server", "Benchmark", "Starlight"}
    server = cands["Inference Server"]
    assert server["source"] == "master_resume+github"
    assert "KV cache" in server["evidence"] and "paged attention" in server["github_evidence"]
    assert "controller/worker" in cands["MedVQA: Medical Data Platform"]["github_evidence"]
    assert cands["Benchmark"]["source"] == "master_resume"
    assert cands["Starlight"]["source"] == "github"


def test_java_certificate_shows_only_on_java_roles():
    from src.render_resume import select_certifications as sel
    certs = [{"name": "Java Full Stack Developer Program", "detail": "Completed",
              "title_include_if": ["java", "spring boot", "jvm", "j2ee", "kotlin"],
              "include_if": ["spring boot", "spring framework", "j2ee", "jvm", "java servlets", "hibernate"]}]

    def guvi(title, desc=""):
        return any("Java Full Stack" in c["name"] for c in sel(certs, f"{title} {desc}", title))

    assert guvi("Java Developer")
    assert guvi("Backend Engineer", "Build microservices with Spring Boot and Kafka")
    assert not guvi("Software Engineer", "Experience in Python, Java or Go")
    assert not guvi("Frontend Engineer", "JavaScript, React")
    assert not guvi("Software Engineer, Spring 2027 Intern")
    assert not guvi("ML Engineer", "PyTorch")


def test_skills_never_repeat():
    import yaml
    from src.render_resume import _dedupe_skills, _reconcile_skills, _split_entries
    groups = [{"category": "Backend", "entries": "FastAPI, Pydantic, LLM APIs (OpenAI, Azure OpenAI)"},
              {"category": "Tools", "entries": "pydantic, Postman"},
              {"category": "Extra", "entries": "Postman"}]
    out = _reconcile_skills(groups, ["Postman", "Pydantic"])
    flat = [e.lower() for g in out for e in _split_entries(g["entries"])]
    assert len(flat) == len(set(flat))
    assert [g["category"] for g in out] == ["Backend", "Tools"]          # emptied group removed
    assert "LLM APIs (OpenAI, Azure OpenAI)" in out[0]["entries"]         # parenthesised commas kept
    # The real master resume lists every skill once.
    mr = yaml.safe_load(open("data/master_resume.yaml"))
    real = [e.lower() for g in mr["skills"] for e in _split_entries(g["entries"])]
    assert len(real) == len(set(real)), [e for e in real if real.count(e) > 1]


def test_skills_tailored_to_the_posting():
    from src.render_resume import tailor_skills
    master = [{"category": "Languages", "entries": "Python, SQL, Java, C#"},
              {"category": "AI / ML", "entries": "PyTorch, diffusion models (DDPM), OpenCV"},
              {"category": "Cloud", "entries": "Docker, Microsoft Azure, Git, Kubernetes"},
              {"category": "Tools", "entries": "Postman, VS Code, Jupyter, Unity3D, Blender"}]
    jd = "Backend engineer: Java, Docker, Kubernetes and Azure; REST services."
    out = tailor_skills(master, ["Java", "Docker", "Postman", "Azure"], jd, core=["Python", "SQL", "Git"])
    cats = [g["category"] for g in out]
    text = " ".join(g["entries"] for g in out)
    assert cats[0] == "Languages" and "AI / ML" not in cats           # irrelevant group hidden
    assert "Microsoft Azure" in text and "Postman" in text            # model picks + posting skills
    assert "Python" in text and "Git" in text                         # core skills kept
    assert "Unity3D" not in text and "DDPM" not in text               # unrelated fields dropped
    assert "C#" in text      # same field: Languages has several relevant skills, so it is filled out
    tools = next(g["entries"] for g in out if g["category"] == "Tools")
    assert tools == "Postman, VS Code, Jupyter"   # 1 relevant -> padded to 3 from the same group
    # Too little left -> keep the full list rather than a near-empty section.
    assert tailor_skills(master, ["Java"], "", core=[]) == master


def test_every_skill_the_posting_names_is_kept():
    from src.render_resume import tailor_skills, _split_entries
    master = [{"category": "Languages", "entries": "Python, SQL, Java, Go"},
              {"category": "LLMs", "entries": "LLM APIs (OpenAI, Azure OpenAI), prompt design"},
              {"category": "Tools", "entries": "Postman, VS Code, Jupyter, Unity3D, Blender"},
              {"category": "Data", "entries": "Pandas, NumPy, SciPy"}]
    jd = ("You will test REST APIs with Postman, call the OpenAI API, and write Python with Pandas "
          "and NumPy. Prompt design a plus.")
    out = tailor_skills(master, [], jd, core=["SQL", "Git"])           # model returned nothing
    kept = [e for g in out for e in _split_entries(g["entries"])]
    for named in ("Postman", "LLM APIs (OpenAI, Azure OpenAI)", "prompt design", "Python"):
        assert named in kept, named
    assert "Unity3D" not in kept
    assert not any("Kubernetes" in e for e in kept)                     # never adds skills you lack


def test_every_shown_group_has_at_least_three():
    from src.render_resume import tailor_skills, _split_entries, MIN_PER_GROUP
    import yaml
    mr = yaml.safe_load(open("data/master_resume.yaml"))
    jd = "Java backend engineer: Spring Boot, PostgreSQL, Docker, REST APIs, Git, Python, Postman."
    out = tailor_skills(mr["skills"], ["Java", "Spring Boot", "PostgreSQL", "Docker"], jd, mr.get("core_skills", []))
    for g in out:
        n = len(_split_entries(g["entries"]))
        full = len(_split_entries(next(m["entries"] for m in mr["skills"] if m["category"] == g["category"])))
        assert n >= min(MIN_PER_GROUP, full), (g["category"], g["entries"])
