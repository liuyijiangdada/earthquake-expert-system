# tests/test_rebuild_toc_yuanyuan_style.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rebuild_toc_yuanyuan_style import is_caption, parse_label


def test_is_caption_figure_ok():
    assert is_caption("图3-1  总体方法框架图", "图")
    assert is_caption("图5-13  分阶段事实一致性对比（各阶段20题）", "图")


def test_is_caption_table_ok_with_space():
    assert is_caption("表 5-1  实验环境配置", "表")
    assert is_caption("表5-2  离线消融实验配置快照", "表")


def test_is_caption_rejects_inline_refs():
    assert not is_caption("如图3-1所示，整体流程分为三阶段", "图")
    assert not is_caption("表5-1给出了实验环境配置", "表")


def test_parse_label_normalizes():
    assert parse_label("图3-1  总体方法框架图", "图") == (3, 1, "图3-1", "总体方法框架图")
    assert parse_label("表 5-1  实验环境配置", "表") == (5, 1, "表5-1", "实验环境配置")
