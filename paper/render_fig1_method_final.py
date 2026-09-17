#!/usr/bin/env python3
"""Camera-ready alias for the three-step MBRS workflow renderer."""

from render_fig1_method import PROJECT, render


if __name__ == "__main__":
    render(
        output_stem="fig1_method_final",
        report_path=PROJECT / "reports" / "fig1_method_final_check.md",
    )
