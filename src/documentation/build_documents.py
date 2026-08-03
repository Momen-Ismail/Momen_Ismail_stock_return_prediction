"""Build all standalone LaTeX documents in documentation/documents."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
SOURCE_DIR = DOCUMENTATION_DIR / "documents"
PDF_DIR = DOCUMENTATION_DIR / "pdf"


OUTPUT_NAMES = {
    "Momen_Ismail": "Momen Ismail",
    "full_project_audit": "full_project_audit",
    "Data_Construction_Guide": "Data_Construction_Guide",
}


def standalone_tex_files():
    """Find complete LaTeX documents, excluding table fragments."""
    documents = []

    for path in sorted(SOURCE_DIR.rglob("*.tex")):
        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if r"\documentclass" in content:
            documents.append(path)

    return documents


def build_document(tex_path):
    """Compile one standalone LaTeX document."""
    if shutil.which("latexmk") is None:
        raise RuntimeError(
            "latexmk is not installed or is not available on PATH."
        )

    output_name = OUTPUT_NAMES.get(
        tex_path.stem,
        tex_path.stem,
    )

    print(f"\nBuilding {tex_path.relative_to(PROJECT_ROOT)}")

    with tempfile.TemporaryDirectory(
        prefix=f"latex_{output_name}_"
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)

        environment = os.environ.copy()

        environment["TEXINPUTS"] = os.pathsep.join([
            str(tex_path.parent),
            str(DOCUMENTATION_DIR / "figures"),
            str(DOCUMENTATION_DIR / "tables"),
            environment.get("TEXINPUTS", ""),
        ])

        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={temporary_directory}",
                tex_path.name,
            ],
            cwd=tex_path.parent,
            env=environment,
            check=True,
        )

        built_pdf = (
            temporary_directory
            / f"{tex_path.stem}.pdf"
        )

        if not built_pdf.exists():
            raise FileNotFoundError(
                f"No PDF was created for {tex_path}."
            )

        PDF_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_pdf = (
            PDF_DIR
            / f"{output_name}.pdf"
        )

        shutil.copy2(
            built_pdf,
            final_pdf,
        )

        print(f"Saved {final_pdf.relative_to(PROJECT_ROOT)}")


def main():
    """Build every complete LaTeX document."""
    documents = standalone_tex_files()

    if not documents:
        raise FileNotFoundError(
            f"No standalone LaTeX documents found under {SOURCE_DIR}."
        )

    for tex_path in documents:
        build_document(tex_path)

    print("\nAll documentation built successfully.")


if __name__ == "__main__":
    main()
