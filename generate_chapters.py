import os
import re
import json

base_path = '/Users/alex/AI_TestPlayground'

def create_notebook(path, title):
    full_path = os.path.join(base_path, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "> 본 노트북은 SYLLABUS.md에 기반하여 자동 생성된 뼈대(Skeleton) 파일입니다. 상세한 이론, 수식 및 코드는 추가로 구현되어야 합니다.\n",
                    "\n",
                    "## 1. 개요 및 학습 목표\n",
                    "이 노트북에서는 해당 주제에 대한 핵심 개념을 다룹니다.\n",
                    "\n",
                    "## 2. 핵심 이론 및 수학적 원리\n",
                    "- 수식 및 상세한 동작 원리를 여기에 기록합니다.\n",
                    "\n",
                    "## 3. 실습 코드 구현\n",
                    "아래 셀을 통해 파이썬 및 관련 프레임워크 코드를 직접 작성하고 실행해 볼 수 있습니다."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 실습을 위한 기본 라이브러리 임포트\n",
                    "import tensorflow as tf\n",
                    "import numpy as np\n",
                    "\n",
                    "print(f\"TensorFlow Version: {tf.__version__}\")"
                ]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

with open(os.path.join(base_path, 'SYLLABUS.md'), 'r', encoding='utf-8') as f:
    text = f.read()

# 정규식을 통해 Chapter 09부터 17까지 매칭
chapters = re.findall(r'## (Chapter (09|1[0-7]) — .*?)\n\*\*디렉토리\*\*: `(.*?)`(.*?)(?=## Chapter|## 최종|\Z)', text, re.DOTALL)

print("Starting notebook generation...")
for chap_title, chap_num, chap_dir, chap_content in chapters:
    chap_dir = chap_dir.strip('/')
    print(f"Processing: {chap_dir}")
    
    # 해당 챕터 안의 .ipynb 파일명과 설명을 매칭
    files = re.findall(r'\| `(.*?\.ipynb)` \| (.*?) \|', chap_content)
    for fname, desc in files:
        file_path = os.path.join(chap_dir, fname)
        create_notebook(file_path, desc)

# 프로젝트 디렉토리 생성 (04~07)
projects = re.findall(r'\| \*\*(.*?)/\*\* \|', text)
for proj in projects:
    proj_dir = proj.strip('`').strip('/')
    if proj_dir.startswith('project04') or proj_dir.startswith('project05') or proj_dir.startswith('project06') or proj_dir.startswith('project07'):
        proj_full = os.path.join(base_path, 'projects', proj_dir)
        os.makedirs(proj_full, exist_ok=True)
        with open(os.path.join(proj_full, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(f"# {proj_dir}\n\nAdvanced Project Folder.\n")
        print(f"Created project dir: {proj_dir}")

print("Successfully generated all skeletons for Chapters 09 to 17.")
