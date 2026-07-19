"""Once-only, zero-external-effect preflight for v3.15 science.

The preflight authenticates the pushed implementation and the read-only v3.8
bridge, freezes aggregate request commitments, runs the frozen qualification suite, and
seals both a private content-addressed manifest and a redacted public artifact.
It never calls SEC, Yahoo, Ollama, a broker, or a paid service.

All effectful dependencies are explicit.  Tests inject synthetic public values;
the command-line adapter lazily connects the local Git, bridge, runner, and
qualification implementation only when the user deliberately runs preflight.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import csv
import copy
import ctypes
from ctypes import wintypes as _wintypes
import dataclasses
import datetime as _datetime
import enum
import hashlib
import heapq
import http.client
import importlib
import importlib.metadata
import importlib.resources
import inspect
import io
import json
import ntpath
import os
from pathlib import Path
import pathlib as _pathlib
import re
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import sysconfig
import threading
import time
import types
import typing as _typing
import urllib.request
import weakref
import zipfile
import zoneinfo
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from collections import Counter
import collections.abc as _collections_abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from . import sec_gemma_lean_science_v315_contract as contract


PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-preflight-v1"
)
PRIVATE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-private-preflight-manifest-v1"
)
PUBLIC_ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-public-preflight-v1"
)
PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-public-preflight-failure-v1"
)
REQUEST_COMMITMENTS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-request-commitments-v1"
)
BRIDGE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-streaming-bridge-v1"
)
PREFLIGHT_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-preflight-intent-v1"
)
QUALIFICATION_REPORT_SCHEMA_VERSION: Final[str] = (
    contract.QUALIFICATION_PUBLIC_RECEIPT_SCHEMA_VERSION
)
QUALIFICATION_COMPLETION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-qualification-completion-v1"
)
EXECUTION_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-execution-authority-v1"
)
PUSHED_RESULT_GATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-pushed-result-gate-v1"
)
TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-terminal-evidence-v1"
)
PUSHED_RESULT_PAUSE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-development-pause-v1"
)
PUSHED_RESULT_RUNTIME_GUARD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-runtime-segment-guard-v1"
)
PUSHED_RESULT_LATENCY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-latency-receipt-v1"
)
REPOSITORY_RUNTIME_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-repository-runtime-manifest-v1"
)
EXECUTION_DEPENDENCY_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-execution-dependency-manifest-v2"
)
LOADED_CODE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-loaded-code-manifest-v2"
)
LAUNCHER_PROFILE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-launcher-profile-v1"
)
RUNTIME_BINDING_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-runtime-binding-authority-v1"
)
RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-runtime-binding-receipt-v1"
)
INTERPRETER_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-interpreter-identity-v1"
)
REPOSITORY_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-repository-state-v1"
)
PREFLIGHT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-preflight-001"
)
PRIVATE_INTENT_FILENAME: Final[str] = "preflight-intent.json"
PRIVATE_MANIFEST_DIRECTORY: Final[str] = "manifests"
PRIVATE_COMPLETION_FILENAME: Final[str] = "preflight-complete.json"
PRIVATE_RUNTIME_DIRECTORY: Final[str] = "runtime"
PRIVATE_RUNTIME_REPOSITORY_FILENAME: Final[str] = (
    "repository_runtime_manifest.json"
)
PRIVATE_RUNTIME_DEPENDENCY_FILENAME: Final[str] = (
    "execution_dependency_manifest.json"
)
PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY: Final[str] = "loaded_code"
PRIVATE_RUNTIME_AUTHORITY_FILENAME: Final[str] = (
    "runtime_binding_authority.json"
)
PRIVATE_CONFIG_PATH: Final[str] = "data/local_config.json"
V38_PRIVATE_ROOT: Final[str] = "data/aapl_sec_gemma_lean_evidence_v3_8"
MAX_PRIVATE_CONFIG_BYTES: Final[int] = 64 * 1024
MAX_RESULT_ARTIFACT_BYTES: Final[int] = 32 * 1024 * 1024
MAX_COMPARISON_BYTES: Final[int] = 4 * 1024 * 1024
_TREE_SCAN_CHUNK_BYTES: Final[int] = 1024 * 1024
_REPARSE_ATTRIBUTE: Final[int] = 0x400
QUALIFICATION_TIMEOUT_SECONDS: Final[int] = contract.QUALIFICATION_TIMEOUT_SECONDS
QUALIFICATION_COMMAND_PROFILE: Final[dict[str, Any]] = {
    "python_flags": list(contract.QUALIFICATION_PYTHON_FLAGS),
    "bootstrap_bytes": contract.QUALIFICATION_BOOTSTRAP_BYTES,
    "bootstrap_sha256": contract.QUALIFICATION_BOOTSTRAP_SHA256,
    "dispatch_prefix": list(contract.QUALIFICATION_DISPATCH_PREFIX),
    "pytest_args": list(contract.QUALIFICATION_PYTEST_ARGS),
    "cwd": "repository_root",
    "environment_names": list(contract.QUALIFICATION_ENVIRONMENT_NAMES),
    "environment_fixed_values": dict(contract.QUALIFICATION_ENVIRONMENT),
    "phases": list(contract.QUALIFICATION_PHASES),
    "modes": list(contract.QUALIFICATION_MODES),
    "stdout": "concurrent_binary_tee",
    "stderr": "concurrent_binary_tee",
    "junit": contract.QUALIFICATION_JUNIT_RELATIVE_TEMPLATE,
    "timeout_seconds": QUALIFICATION_TIMEOUT_SECONDS,
    "network_authorized": False,
}
QUALIFICATION_COMMAND_PROFILE_SHA256: Final[str] = contract.canonical_sha256(
    QUALIFICATION_COMMAND_PROFILE
)

# Exact stdlib-only scientific launcher.  It validates the isolated process,
# authenticates P->I for creation or pushed F for consumption, installs the
# process-local proof, and only then imports a canonical V3.15 entry point.
# Direct ``python -m`` execution never installs either proof and fails closed.
SCIENTIFIC_BOOTSTRAP_LITERAL: Final[str] = r'''import builtins,ctypes,hashlib,json,ntpath,os,stat,subprocess,sys,sysconfig
from ctypes import wintypes

def fail(code):
    raise SystemExit(code)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_bytes(path,require_single_link=False):
    before=os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or getattr(before,'st_file_attributes',0)&0x400 or before.st_nlink<1 or require_single_link and before.st_nlink!=1:
        fail('bootstrap_file_invalid')
    with open(path,'rb') as handle:
        opened_before=os.fstat(handle.fileno())
        payload=handle.read()
        opened_after=os.fstat(handle.fileno())
    after=os.lstat(path)
    identity=lambda value:(value.st_dev,value.st_ino,value.st_nlink,value.st_size,value.st_mtime_ns,getattr(value,'st_file_attributes',0))
    descriptor=lambda value:(*identity(value),value.st_mode,value.st_ctime_ns)
    if any(not stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode) or getattr(value,'st_file_attributes',0)&0x400 or value.st_nlink<1 for value in (opened_before,opened_after,after)) or require_single_link and any(value.st_nlink!=1 for value in (opened_before,opened_after,after)) or identity(before)!=identity(opened_before) or descriptor(opened_before)!=descriptor(opened_after) or identity(opened_after)!=identity(after) or any(value.st_size!=len(payload) for value in (before,opened_before,opened_after,after)):
        fail('bootstrap_file_race')
    return payload

def path_token(path):
    resolved=os.path.realpath(path,strict=True)
    value=ntpath.normcase(ntpath.normpath(resolved)).replace('/','\\')
    drive,tail=ntpath.splitdrive(value)
    if value.endswith('\\') and tail not in ('\\',''):
        value=value.rstrip('\\')
    return hashlib.sha256(b'aapl-v315-private-windows-path-v1\x00'+value.encode('utf-8')).hexdigest()

flag_names=('debug','inspect','interactive','optimize','dont_write_bytecode','no_user_site','no_site','ignore_environment','verbose','bytes_warning','quiet','hash_randomization','isolated','dev_mode','utf8_mode','warn_default_encoding','safe_path','int_max_str_digits')
flag_values=(0,0,0,0,1,1,1,0,0,0,0,0,0,False,1,0,False,4300)
if tuple(getattr(sys.flags,name) for name in flag_names)!=flag_values:
    fail('bootstrap_flags_invalid')
fixed={'PYTHONNOUSERSITE':'1','PYTHONHASHSEED':'0','PYTHONPATH':'','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','TZ':'America/New_York','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
env_names=tuple(sorted(('SYSTEMROOT','WINDIR','TEMP','TMP','PATH',*fixed)))
if tuple(sorted(os.environ))!=env_names or any(os.environ.get(key)!=value for key,value in fixed.items()) or any(not os.environ.get(key) for key in ('SYSTEMROOT','WINDIR','TEMP','TMP','PATH')):
    fail('bootstrap_environment_invalid')
if any(name in sys.modules for name in ('site','sitecustomize','usercustomize')):
    fail('bootstrap_customization_loaded')

exe=os.path.realpath(sys.executable,strict=True)
parent=os.path.dirname(exe)
zip_path=os.path.join(parent,'python312.zip')
initial=('',zip_path,os.path.join(parent,'DLLs'),os.path.join(parent,'Lib'),parent)
key=lambda path:ntpath.normcase(ntpath.normpath(path))
if len(sys.path)!=5 or sys.path[0]!='' or any(key(left)!=key(right) for left,right in zip(sys.path,initial)) or os.path.lexists(zip_path) or any(not os.path.isdir(path) for path in initial[2:]):
    fail('bootstrap_initial_sys_path_invalid')
if len(sys.argv)<5 or sys.argv[1]!='--repo-root' or sys.argv[3]!='--':
    fail('bootstrap_argv_invalid')
root=os.path.realpath(sys.argv[2],strict=True)
route=sys.argv[4]
extra=sys.argv[5:]
if not os.path.isdir(root) or key(root)!=key(os.path.abspath(sys.argv[2])) or getattr(os.lstat(root),'st_file_attributes',0)&0x400:
    fail('bootstrap_root_invalid')

ordered=[]
for candidate in (root,initial[2],initial[3],initial[4],sysconfig.get_path('purelib'),sysconfig.get_path('platlib')):
    resolved=os.path.realpath(candidate,strict=True)
    if not os.path.isdir(resolved):
        fail('bootstrap_path_invalid')
    if key(resolved) not in {key(item) for item in ordered}:
        ordered.append(resolved)
sys.path[:]=ordered
final_tokens=[path_token(path) for path in ordered]
final_path_sha256=digest(final_tokens)

path_parts=[]
for item in os.environ['PATH'].split(';'):
    resolved=os.path.realpath(item,strict=True)
    if key(resolved) not in {key(value) for value in path_parts}:
        path_parts.append(resolved)
git_exe=os.path.realpath(os.path.join(path_parts[0],'git.exe'),strict=True)
expected_path=[]
for candidate in (os.path.dirname(git_exe),os.path.dirname(exe),os.path.join(os.environ['SYSTEMROOT'],'System32'),os.environ['SYSTEMROOT']):
    resolved=os.path.realpath(candidate,strict=True)
    if key(resolved) not in {key(value) for value in expected_path}:
        expected_path.append(resolved)
if [key(value) for value in path_parts]!=[key(value) for value in expected_path]:
    fail('bootstrap_path_environment_invalid')
environment_material={name:(path_token(os.environ[name]) if name in ('SYSTEMROOT','WINDIR','TEMP','TMP') else [path_token(value) for value in path_parts] if name=='PATH' else os.environ[name]) for name in env_names}
process_environment_sha256=digest(environment_material)

kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
shell32=ctypes.WinDLL('shell32',use_last_error=True)
kernel32.GetCommandLineW.argtypes=[]
kernel32.GetCommandLineW.restype=wintypes.LPWSTR
shell32.CommandLineToArgvW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int)]
shell32.CommandLineToArgvW.restype=ctypes.POINTER(wintypes.LPWSTR)
kernel32.LocalFree.argtypes=[wintypes.HLOCAL]
kernel32.LocalFree.restype=wintypes.HLOCAL
argc=ctypes.c_int()
argvp=shell32.CommandLineToArgvW(kernel32.GetCommandLineW(),ctypes.byref(argc))
if not argvp:
    fail('bootstrap_command_line_invalid')
try:
    raw_argv=[argvp[index] for index in range(argc.value)]
finally:
    kernel32.LocalFree(argvp)
positions=[index for index,value in enumerate(raw_argv[:-1]) if value=='-c']
if len(positions)!=1:
    fail('bootstrap_command_line_invalid')
bootstrap_bytes=raw_argv[positions[0]+1].encode('utf-8')
bootstrap_sha256=hashlib.sha256(bootstrap_bytes).hexdigest()

python_payload=file_bytes(exe)
if os.path.basename(exe)!='python.exe' or len(python_payload)!=103192 or hashlib.sha256(python_payload).hexdigest()!='624bbc0586d8855633b875e911883bbef8a0e8b8711e11126df480dd86f54181' or sys.version!='3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)]' or sys.implementation.cache_tag!='cpython-312':
    fail('bootstrap_python_invalid')
git_payload=file_bytes(git_exe)
git_env=dict(os.environ)
git_env.update({'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never','GIT_OPTIONAL_LOCKS':'0'})
def git(*args,binary=False):
    result=subprocess.run([git_exe,'-C',root,*args],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=git_env,timeout=30,check=False)
    if result.returncode!=0:
        fail('bootstrap_git_invalid')
    if binary:
        return result.stdout
    try:
        return result.stdout.decode('utf-8','strict').strip()
    except UnicodeError:
        fail('bootstrap_git_invalid')
git_version=subprocess.run([git_exe,'--version'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=git_env,timeout=30,check=False)
if git_version.returncode!=0 or not git_version.stdout.startswith(b'git version '):
    fail('bootstrap_git_invalid')
runtime_root=os.path.join(root,'data','aapl_sec_gemma_lean_science_v3_15','preflight','runtime')
dependency_path=os.path.join(runtime_root,'execution_dependency_manifest.json')
authority_path=os.path.join(runtime_root,'runtime_binding_authority.json')
if route!='preflight':
    try:
        dependency_payload=file_bytes(dependency_path,True)
        dependency=json.loads(dependency_payload.decode('utf-8','strict'))
        authority_payload=file_bytes(authority_path,True)
        runtime_authority=json.loads(authority_payload.decode('utf-8','strict'))
        public_working=file_bytes(os.path.join(root,'e','aapl_sec_gemma_lean_science_v3_15','DEVELOPMENT_PREFLIGHT.json'),True)
        public_value=json.loads(public_working.decode('utf-8','strict'))
    except (OSError,UnicodeError,ValueError):
        fail('bootstrap_runtime_authority_invalid')
    if canonical(dependency)!=dependency_payload or canonical(runtime_authority)!=authority_payload or canonical(public_value)+b'\n'!=public_working:
        fail('bootstrap_runtime_authority_invalid')
    dep_unsigned=dict(dependency)
    dep_hash=dep_unsigned.pop('execution_dependency_manifest_sha256',None)
    auth_unsigned=dict(runtime_authority)
    auth_hash=auth_unsigned.pop('runtime_binding_authority_sha256',None)
    if dep_hash!=digest(dep_unsigned) or auth_hash!=digest(auth_unsigned) or runtime_authority.get('execution_dependency_manifest_sha256')!=dep_hash or public_value.get('runtime_binding_authority_sha256')!=auth_hash:
        fail('bootstrap_runtime_authority_invalid')
    git_row=dependency.get('git_runtime',{})
    python_row=dependency.get('python_runtime',{})
    if git_row.get('resolved_path_sha256')!=path_token(git_exe) or git_row.get('byte_count')!=len(git_payload) or git_row.get('literal_sha256')!=hashlib.sha256(git_payload).hexdigest() or git_row.get('version_output_sha256')!=hashlib.sha256(git_version.stdout).hexdigest() or python_row.get('executable_path_sha256')!=path_token(exe) or python_row.get('executable_bytes')!=len(python_payload) or python_row.get('executable_sha256')!=hashlib.sha256(python_payload).hexdigest() or python_row.get('bootstrap_sha256')!=bootstrap_sha256 or python_row.get('process_environment_sha256')!=process_environment_sha256 or dependency.get('sys_path_sha256')!=final_path_sha256:
        fail('bootstrap_runtime_identity_invalid')
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel32.GetCurrentProcess.argtypes=[]
    kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.EnumProcessModules.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.HMODULE),wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    psapi.EnumProcessModules.restype=wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes=[wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype=wintypes.DWORD
    process=kernel32.GetCurrentProcess()
    images=(wintypes.HMODULE*512)()
    needed=wintypes.DWORD()
    if not psapi.EnumProcessModules(process,images,ctypes.sizeof(images),ctypes.byref(needed)) or needed.value>ctypes.sizeof(images):
        fail('bootstrap_python_dll_invalid')
    dll_paths=[]
    for handle in images[:needed.value//ctypes.sizeof(wintypes.HMODULE)]:
        buffer=ctypes.create_unicode_buffer(32768)
        length=psapi.GetModuleFileNameExW(process,handle,buffer,len(buffer))
        if length and os.path.basename(buffer.value).lower()=='python312.dll':
            dll_paths.append(os.path.realpath(buffer.value,strict=True))
    if len(dll_paths)!=1:
        fail('bootstrap_python_dll_invalid')
    dll_payload=file_bytes(dll_paths[0])
    if python_row.get('python_dll_path_sha256')!=path_token(dll_paths[0]) or python_row.get('python_dll_bytes')!=len(dll_payload) or python_row.get('python_dll_sha256')!=hashlib.sha256(dll_payload).hexdigest():
        fail('bootstrap_python_dll_invalid')
branch=git('branch','--show-current')
upstream=git('rev-parse','--abbrev-ref','--symbolic-full-name','@{upstream}')
origin=git('remote','get-url','origin')
head=git('rev-parse','HEAD')
remote=git('rev-parse','refs/remotes/origin/codex/aapl-sec-gemma-lean-science-v3-15')
if branch!='codex/aapl-sec-gemma-lean-science-v3-15' or upstream!='origin/codex/aapl-sec-gemma-lean-science-v3-15' or origin!='https://github.com/AntonioDomenech/LLM-memory-trading-agent.git' or head!=remote:
    fail('bootstrap_repository_identity_invalid')
def parent_of(commit):
    value=git('rev-list','--parents','-n','1',commit).split()
    if len(value)!=2 or value[0]!=commit:
        fail('bootstrap_ancestry_invalid')
    return value[1]
def delta(parent,child):
    text=git('diff-tree','--no-commit-id','--name-status','-r',parent,child)
    return text.splitlines() if text else []
paths=('agent_benchmark/sec_gemma_lean_science_v315_contract.py','agent_benchmark/sec_gemma_lean_science_v315_bridge.py','agent_benchmark/sec_gemma_lean_science_v315_journal.py','agent_benchmark/sec_gemma_lean_science_v315_store.py','agent_benchmark/sec_gemma_lean_science_v315_preflight.py','agent_benchmark/sec_gemma_lean_science_v315_runner.py','tests/test_sec_gemma_lean_science_v315_contract.py','tests/test_sec_gemma_lean_science_v315_bridge.py','tests/test_sec_gemma_lean_science_v315_journal.py','tests/test_sec_gemma_lean_science_v315_store.py','tests/test_sec_gemma_lean_science_v315_preflight.py','tests/test_sec_gemma_lean_science_v315_runner.py')
if route=='preflight':
    if extra or git('status','--porcelain=v1','--untracked-files=all')!='':
        fail('bootstrap_creation_state_invalid')
    implementation=head
    preregistration=parent_of(implementation)
    if preregistration!='3955520674a3b6ccfa29cf9fc6593e110917e437' or parent_of(preregistration)!='3e813577f1aab4ae80d0f5f59362a46320c818fb' or git('rev-parse',preregistration+'^{tree}')!='c07f5893b235b145914cb115e1888fb360d59b34' or git('rev-parse','3e813577f1aab4ae80d0f5f59362a46320c818fb^{tree}')!='90d4e6681acb81fc11f5abf9860df331923a907d' or parent_of('3e813577f1aab4ae80d0f5f59362a46320c818fb')!='a6b5346e8bf6edec66fad38d35392a6d0a2c513b':
        fail('bootstrap_creation_ancestry_invalid')
    if delta(preregistration,implementation)!=['A\t'+path for path in sorted(paths)]:
        fail('bootstrap_creation_delta_invalid')
    if git('rev-parse',preregistration+':docs/aapl_sec_gemma_lean_science_v3_15.md')!='0d00607ac6a8fc548987c038e951f0c831437097':
        fail('bootstrap_preregistration_invalid')
    prereg_payload=git('show',preregistration+':docs/aapl_sec_gemma_lean_science_v3_15.md',binary=True)
    if len(prereg_payload)!=24926 or hashlib.sha256(prereg_payload).hexdigest()!='48af36ce45277be93fdb0af6e2a5173be725f8305f03ad956dec8e32cce37d65':
        fail('bootstrap_preregistration_invalid')
    source=[]
    for relative in paths:
        committed=git('show',implementation+':'+relative,binary=True)
        working=file_bytes(os.path.join(root,*relative.split('/')),True)
        blob=git('rev-parse',implementation+':'+relative)
        if committed!=working:
            fail('bootstrap_source_invalid')
        sha=hashlib.sha256(committed).hexdigest()
        source.append({'path':relative,'git_blob_sha1':blob,'git_literal_sha256':sha,'git_byte_count':len(committed),'working_literal_sha256':sha,'working_byte_count':len(working)})
    body={'schema_version':'aapl-sec-gemma-lean-science-v3-15-creation-pretrust-attestation-v1','base_commit':'3e813577f1aab4ae80d0f5f59362a46320c818fb','base_tree':'90d4e6681acb81fc11f5abf9860df331923a907d','preregistration_commit':preregistration,'preregistration_tree':'c07f5893b235b145914cb115e1888fb360d59b34','preregistration_git_blob_sha1':'0d00607ac6a8fc548987c038e951f0c831437097','preregistration_literal_sha256':'48af36ce45277be93fdb0af6e2a5173be725f8305f03ad956dec8e32cce37d65','preregistration_literal_bytes':24926,'implementation_commit':implementation,'implementation_tree':git('rev-parse',implementation+'^{tree}'),'implementation_parent':preregistration,'branch':branch,'upstream':upstream,'origin_url':origin,'repository_root_sha256':path_token(root),'source_inventory':source,'source_inventory_sha256':digest(source),'clean_worktree':True,'git_runtime':{'basename':os.path.basename(git_exe),'resolved_path_sha256':path_token(git_exe),'byte_count':len(git_payload),'literal_sha256':hashlib.sha256(git_payload).hexdigest(),'version_output_sha256':hashlib.sha256(git_version.stdout).hexdigest()},'python_runtime':{'basename':os.path.basename(exe),'resolved_path_sha256':path_token(exe),'byte_count':len(python_payload),'literal_sha256':hashlib.sha256(python_payload).hexdigest(),'python_version':sys.version,'cache_tag':sys.implementation.cache_tag},'bootstrap':{'byte_count':len(bootstrap_bytes),'literal_sha256':bootstrap_sha256},'effect_counts':{'sec_requests':0,'experiment_family_sec_requests':964,'yahoo_requests':0,'ollama_identity_http_requests':0,'ollama_chat_generations':0,'retries':0,'repairs':0,'pulls':0,'fallbacks':0,'paid_calls':0,'confirmation_final_data_opens':0,'broker_effects':0,'real_money_effects':0}}
    body['attestation_sha256']=digest(body)
    setattr(builtins,'_AAPL_SEC_GEMMA_LEAN_SCIENCE_V315_CREATION_PRETRUST',body)
    module=__import__('agent_benchmark.sec_gemma_lean_science_v315_preflight',fromlist=('main',))
    raise SystemExit(module.main([root]))

if route not in ('development','continue-development','recover-publication'):
    fail('bootstrap_route_invalid')
if route!='continue-development' and extra:
    fail('bootstrap_route_arguments_invalid')
if route=='continue-development' and (len(extra)!=1 or len(extra[0])!=64 or any(character not in '0123456789abcdef' for character in extra[0])):
    fail('bootstrap_continuation_permission_invalid')
invocation={'development':'development','continue-development':'continuation','recover-publication':'publication_recovery'}[route]
status_z=git('status','--porcelain=v1','-z','--untracked-files=all',binary=True)
if route!='recover-publication' and status_z!=b'':
    fail('bootstrap_consumption_state_invalid')
if route=='development':
    preflight_commit=head
elif route=='continue-development':
    pause_commit=parent_of(head)
    preflight_commit=parent_of(pause_commit)
    if delta(pause_commit,head)!=['A\tdocs/aapl_sec_gemma_lean_science_v3_15_continuation.md'] or delta(preflight_commit,pause_commit)!=['A\te/aapl_sec_gemma_lean_science_v3_15/DEVELOPMENT_PAUSE.json']:
        fail('bootstrap_continuation_ancestry_invalid')
else:
    parent=parent_of(head)
    direct=delta(parent,head)
    if direct==['A\te/aapl_sec_gemma_lean_science_v3_15/DEVELOPMENT_PREFLIGHT.json']:
        preflight_commit=head
    elif direct==['A\tdocs/aapl_sec_gemma_lean_science_v3_15_continuation.md']:
        pause_commit=parent
        preflight_commit=parent_of(pause_commit)
        if delta(preflight_commit,pause_commit)!=['A\te/aapl_sec_gemma_lean_science_v3_15/DEVELOPMENT_PAUSE.json']:
            fail('bootstrap_recovery_ancestry_invalid')
    else:
        fail('bootstrap_recovery_ancestry_invalid')
implementation=parent_of(preflight_commit)
if parent_of(implementation)!='3955520674a3b6ccfa29cf9fc6593e110917e437' or delta(implementation,preflight_commit)!=['A\te/aapl_sec_gemma_lean_science_v3_15/DEVELOPMENT_PREFLIGHT.json']:
    fail('bootstrap_preflight_ancestry_invalid')
public_relative='e/aapl_sec_gemma_lean_science_v3_15/DEVELOPMENT_PREFLIGHT.json'
public_payload=git('show',preflight_commit+':'+public_relative,binary=True)
if file_bytes(os.path.join(root,*public_relative.split('/')),True)!=public_payload:
    fail('bootstrap_preflight_public_invalid')
runtime_root=os.path.join(root,'data','aapl_sec_gemma_lean_science_v3_15','preflight','runtime')
repository_path=os.path.join(runtime_root,'repository_runtime_manifest.json')
authority_path=os.path.join(runtime_root,'runtime_binding_authority.json')
try:
    repository_payload=file_bytes(repository_path,True)
    repository_manifest=json.loads(repository_payload.decode('utf-8','strict'))
    authority_payload=file_bytes(authority_path,True)
    runtime_authority=json.loads(authority_payload.decode('utf-8','strict'))
except (OSError,UnicodeError,ValueError):
    fail('bootstrap_runtime_authority_invalid')
if canonical(repository_manifest)!=repository_payload or canonical(runtime_authority)!=authority_payload:
    fail('bootstrap_runtime_authority_invalid')
repository_unsigned=dict(repository_manifest)
repository_hash=repository_unsigned.pop('repository_runtime_manifest_sha256',None)
authority_unsigned=dict(runtime_authority)
authority_hash=authority_unsigned.pop('runtime_binding_authority_sha256',None)
if repository_hash!=digest(repository_unsigned) or authority_hash!=digest(authority_unsigned) or runtime_authority.get('repository_runtime_manifest_sha256')!=repository_hash or repository_manifest.get('implementation_commit')!=implementation or repository_manifest.get('implementation_tree')!=git('rev-parse',implementation+'^{tree}') or repository_manifest.get('module_count')!=44 or not isinstance(repository_manifest.get('modules'),list) or len(repository_manifest['modules'])!=44:
    fail('bootstrap_runtime_authority_invalid')
v315_paths=sorted(row.get('relative_path') for row in repository_manifest['modules'] if row.get('role')=='v315')
shared_paths=sorted(row.get('relative_path') for row in repository_manifest['modules'] if row.get('role')=='shared')
package_paths=sorted(row.get('relative_path') for row in repository_manifest['modules'] if row.get('role')=='package_bootstrap')
if v315_paths!=sorted(paths[:6]) or len(shared_paths)!=37 or digest(shared_paths)!='0a1e7374b35677596418cfa82575bd6614df631eba34f1978ce9c9017a74aae9' or package_paths!=['agent_benchmark/__init__.py']:
    fail('bootstrap_runtime_repository_allowlist_invalid')
for row in repository_manifest['modules']:
    if set(row)!=set(('module_name','relative_path','role','owning_revision','git_blob_sha1','literal_sha256','byte_count')):
        fail('bootstrap_runtime_repository_invalid')
    relative=row['relative_path']
    if not isinstance(relative,str) or relative.startswith(('/', '../')) or '\\' in relative or ntpath.splitdrive(relative)[0] or any(part in ('','.','..') for part in relative.split('/')):
        fail('bootstrap_runtime_repository_invalid')
    target=os.path.realpath(os.path.join(root,*relative.split('/')),strict=True)
    if key(os.path.commonpath((root,target)))!=key(root):
        fail('bootstrap_runtime_repository_invalid')
    expected_module=relative[:-len('/__init__.py')].replace('/','.') if relative.endswith('/__init__.py') else relative[:-3].replace('/','.')
    expected_owner=implementation if row['role']=='v315' else '3e813577f1aab4ae80d0f5f59362a46320c818fb'
    if row['module_name']!=expected_module or row['owning_revision']!=expected_owner:
        fail('bootstrap_runtime_repository_invalid')
    working=file_bytes(target,True)
    committed=git('show',row['owning_revision']+':'+relative,binary=True)
    implementation_bytes=git('show',implementation+':'+relative,binary=True)
    if working!=committed or working!=implementation_bytes or len(working)!=row['byte_count'] or hashlib.sha256(working).hexdigest()!=row['literal_sha256'] or git('rev-parse',row['owning_revision']+':'+relative)!=row['git_blob_sha1'] or git('rev-parse',implementation+':'+relative)!=row['git_blob_sha1']:
        fail('bootstrap_runtime_repository_invalid')
launch={'schema_version':'aapl-sec-gemma-lean-science-v3-15-runtime-launch-v1','invocation_kind':invocation,'repository_root_sha256':path_token(root),'bootstrap_sha256':bootstrap_sha256,'bootstrap_bytes':len(bootstrap_bytes),'process_environment_sha256':process_environment_sha256,'final_sys_path_sha256':final_path_sha256}
launch['runtime_launch_sha256']=digest(launch)
setattr(builtins,'_AAPL_SEC_GEMMA_LEAN_SCIENCE_V315_RUNTIME_LAUNCH',launch)
runner=__import__('agent_benchmark.sec_gemma_lean_science_v315_runner',fromlist=('main',))
arguments=['development' if route!='recover-publication' else 'recover-publication','--repo-root',root]
if route=='continue-development':
    arguments.extend(('--continuation-permission-sha256',extra[0]))
raise SystemExit(runner.main(arguments))'''
SCIENTIFIC_BOOTSTRAP_BYTES: Final[bytes] = SCIENTIFIC_BOOTSTRAP_LITERAL.encode(
    "utf-8"
)
SCIENTIFIC_BOOTSTRAP_SHA256: Final[str] = hashlib.sha256(
    SCIENTIFIC_BOOTSTRAP_BYTES
).hexdigest()

_SHA1_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE: Final[re.Pattern[bytes]] = re.compile(
    rb"[0-9]{10}-[0-9]{2}-[0-9]{6}"
)
_PRIVATE_PATH_DOMAIN: Final[bytes] = b"aapl-v315-private-windows-path-v1"
CREATION_PRETRUST_ATTESTATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-creation-pretrust-attestation-v1"
)
CREATION_PRETRUST_SENTINEL_NAME: Final[str] = (
    "_AAPL_SEC_GEMMA_LEAN_SCIENCE_V315_CREATION_PRETRUST"
)
SCIENTIFIC_RUNTIME_SENTINEL_NAME: Final[str] = (
    "_AAPL_SEC_GEMMA_LEAN_SCIENCE_V315_RUNTIME_LAUNCH"
)
CREATION_PRETRUST_EXPECTED_ORIGIN_URL: Final[str] = (
    "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
)
_CREATION_PRETRUST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "base_commit",
        "base_tree",
        "preregistration_commit",
        "preregistration_tree",
        "preregistration_git_blob_sha1",
        "preregistration_literal_sha256",
        "preregistration_literal_bytes",
        "implementation_commit",
        "implementation_tree",
        "implementation_parent",
        "branch",
        "upstream",
        "origin_url",
        "repository_root_sha256",
        "source_inventory",
        "source_inventory_sha256",
        "clean_worktree",
        "git_runtime",
        "python_runtime",
        "bootstrap",
        "effect_counts",
        "attestation_sha256",
    }
)
_CREATION_PRETRUST_SOURCE_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "git_blob_sha1",
        "git_literal_sha256",
        "git_byte_count",
        "working_literal_sha256",
        "working_byte_count",
    }
)
_CREATION_PRETRUST_GIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "basename",
        "resolved_path_sha256",
        "byte_count",
        "literal_sha256",
        "version_output_sha256",
    }
)
_CREATION_PRETRUST_PYTHON_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "basename",
        "resolved_path_sha256",
        "byte_count",
        "literal_sha256",
        "python_version",
        "cache_tag",
    }
)
_CREATION_PRETRUST_BOOTSTRAP_FIELDS: Final[frozenset[str]] = frozenset(
    {"byte_count", "literal_sha256"}
)
RUNTIME_CHECKPOINT_NAMES: Final[tuple[str, ...]] = (
    "attempt_open",
    "yahoo_01_pre",
    "yahoo_02_pre",
    "yahoo_03_pre",
    "yahoo_04_pre",
    "yahoo_05_pre",
    "yahoo_06_pre",
    "model_initial_pre",
    "model_initial_post",
    "pause_publish_pre",
    "model_continuation_pre",
    "model_continuation_post",
    "evaluation_pre",
    "evaluation_post",
    "result_publish_pre",
    "publication_recovery_pre",
)
RUNTIME_ROUTE_ARGUMENTS: Final[dict[str, list[str]]] = {
    "development": ["development"],
    "continuation": ["continue-development"],
    "publication_recovery": ["recover-publication"],
}
RUNTIME_ENVIRONMENT_ALLOWLIST: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PATH",
            "PYTHONNOUSERSITE",
            "PYTHONHASHSEED",
            "PYTHONPATH",
            "PYTHONUTF8",
            "PYTHONIOENCODING",
            "TZ",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ),
        key=lambda name: name.encode("utf-8"),
    )
)
_FIXED_RUNTIME_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPATH": "",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "TZ": "America/New_York",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
_RUNTIME_SUCCESSOR_BASE_COMMIT: Final[str] = contract.BASE_COMMIT
_RUNTIME_CALLABLE_ALGORITHM: Final[str] = (
    "python312_recursive_callable_namespace_v2"
)
_RUNTIME_TIMEZONE_KEYS: Final[tuple[str, ...]] = (
    "America/Chicago",
    "America/New_York",
)
_RUNTIME_PSEUDO_MODULE_ALIAS_SPECS: Final[
    tuple[tuple[str, str, str, str, str], ...]
] = (
    ("pyexpat.errors", "pyexpat", "errors", "module", "builtins.module"),
    ("pyexpat.model", "pyexpat", "model", "module", "builtins.module"),
    (
        "typing.io",
        "typing",
        "io",
        "deprecated_type",
        "typing._DeprecatedType",
    ),
    (
        "typing.re",
        "typing",
        "re",
        "deprecated_type",
        "typing._DeprecatedType",
    ),
)
_RUNTIME_RELATIVE_PATH_BINDINGS: Final[
    tuple[tuple[str, str, str], ...]
] = (
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_runtime",
        "MANIFEST_RELATIVE_PATH",
        "manifests/registry.ollama.ai/library/gemma4/12b",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_store",
        "STATE_RELATIVE_DIRECTORY",
        "data/sec_gemma_online_risk_overlay_v2_2/64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_store",
        "ANCHOR_RELATIVE_DIRECTORY",
        "data/sec_gemma_online_risk_overlay_v2_2_anchors",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_vault",
        "PRODUCTION_VAULT_RELATIVE_PATH",
        "data/sec_gemma_online_risk_overlay_v2_2/64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d/quarantine.sqlite3",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_vault",
        "STATE_RELATIVE_DIRECTORY",
        "data/sec_gemma_online_risk_overlay_v2_2/64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d",
    ),
)
_RUNTIME_TYPING_ALIAS_BINDINGS: Final[
    tuple[tuple[str, str, str], ...]
] = (
    ("agent_benchmark.sec_audit_transport", "Mapping", "Mapping"),
    ("agent_benchmark.sec_filing_content", "Mapping", "Mapping"),
    ("agent_benchmark.sec_gemma_lean_v38_source", "Mapping", "Mapping"),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_attempt",
        "Mapping",
        "Mapping",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_publisher",
        "Mapping",
        "Mapping",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_registry",
        "Mapping",
        "Mapping",
    ),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier",
        "Mapping",
        "Mapping",
    ),
    ("agent_benchmark.sec_point_in_time", "Mapping", "Mapping"),
    ("agent_benchmark.sec_filing_content", "Sequence", "Sequence"),
    ("agent_benchmark.sec_gemma_lean_v38_source", "Sequence", "Sequence"),
    (
        "agent_benchmark.sec_gemma_online_risk_overlay_attempt",
        "Sequence",
        "Sequence",
    ),
    ("agent_benchmark.sec_point_in_time", "Sequence", "Sequence"),
    ("agent_benchmark.sec_session_calendar", "Sequence", "Sequence"),
)
_RUNTIME_OWNER_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "class_base",
        "class_method",
        "class_staticmethod",
        "class_classmethod",
        "class_property_fget",
        "class_property_fset",
        "class_property_fdel",
        "class_nested_class",
        "class_annotation",
        "function_default",
        "function_kwdefault",
        "function_annotation",
        "function_closure",
        "function_referenced_global",
        "list_item",
        "tuple_item",
        "mapping_value",
        "dataclass_field",
        "dataclass_type",
        "enum_type",
        "weak_registry_remove",
        "weak_registry_self_ref",
        "relative_path_type",
        "typing_alias_type",
        "typing_alias_origin",
    }
)
_RUNTIME_REQUIRED_PHASE1_NODE_IDS: Final[tuple[str, ...]] = (
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_phase2_case_sensitive_multiplicity_is_655_unique_zero_duplicates",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_phase2_rejects_case_insensitive_node_id_collapsing",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_isolated_qualifier_bootstrap_matches_frozen_literal_and_imports_pinned_pytest",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_authority_creation_precedes_authority_consumption",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_authority_creation_never_requires_f_or_pushed_authority",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_development_consumes_only_pushed_gated_authority",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_continuation_authorized_precedes_attempt_open_binding",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_continuation_attempt_open_precedes_model_continuation_pre",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_model_continuation_pre_precedes_first_ollama_intent",
    "tests/test_sec_gemma_lean_science_v315_store.py::test_recovery_api_is_reachable_end_to_end_with_ordered_demotion_and_verification_only",
    "tests/test_sec_gemma_lean_science_v315_store.py::test_nested_mappingproxytype_evidence_is_recursively_thawed_and_canonicalized",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_forbidden_old_validator_sentinel_is_retained_and_triggered",
    "tests/test_sec_gemma_lean_science_v315_runner.py::test_pushed_result_gate_uses_candidate_material_schema_with_plan_equals_plan",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_owner_graph_has_no_discovered_pseudo_slots_or_unrooted_callables",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_owner_slots_disambiguate_every_same_natural_callable_identity",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_closure_state_allowlist_binds_exact_five_objects_and_slots",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_weak_registries_are_empty_callback_bound_and_identity_continuous",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_relative_path_allowlist_is_exact_and_alias_bound",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_typing_alias_allowlist_is_exact_and_identity_bound",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_runtime_pseudo_module_normalization_requires_all_four_once",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_loaded_code_v2_callable_rows_bind_owner_slot_hashes",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_real_v315_production_import_calls_are_literal_and_upper_bound_closes",
    "tests/test_sec_gemma_lean_science_v315_preflight.py::test_junit_counts_accepts_exact_pytest_long_duration_clock_suffix",
)

# Process-local mutable state lives outside every sealed repository namespace.
# The module retains only one literal identity sentinel; replacement of that
# sentinel is detected by loaded-code identity continuity.
_RUNTIME_PROCESS_STATE_SENTINEL: Final[object] = object()
_RUNTIME_PROCESS_STATE_NAME: Final[str] = (
    "_AAPL_SEC_GEMMA_LEAN_SCIENCE_V315_PROCESS_STATE"
)
_RUNTIME_LOCK_TYPE: Final[type] = type(threading.Lock())
_RUNTIME_RLOCK_TYPE: Final[type] = type(threading.RLock())


def _runtime_process_state() -> dict[str, Any]:
    value = getattr(builtins, _RUNTIME_PROCESS_STATE_NAME, None)
    if value is None:
        value = {
            "sentinel": _RUNTIME_PROCESS_STATE_SENTINEL,
            "active_preflight": {},
            "binding_sessions": {},
            "pseudo_module_normalization": None,
        }
        setattr(builtins, _RUNTIME_PROCESS_STATE_NAME, value)
    if (
        type(value) is not dict
        or value.get("sentinel") is not _RUNTIME_PROCESS_STATE_SENTINEL
        or type(value.get("active_preflight")) is not dict
        or type(value.get("binding_sessions")) is not dict
        or (
            value.get("pseudo_module_normalization") is not None
            and type(value.get("pseudo_module_normalization")) is not dict
        )
        or set(value)
        != {
            "sentinel",
            "active_preflight",
            "binding_sessions",
            "pseudo_module_normalization",
        }
    ):
        _reject("runtime_process_state_invalid")
    return value


def _runtime_active_preflight_context() -> dict[str, Any]:
    return _runtime_process_state()["active_preflight"]


def _runtime_binding_sessions() -> dict[str, dict[str, Any]]:
    return _runtime_process_state()["binding_sessions"]

_REPOSITORY_RUNTIME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "branch",
        "origin_url",
        "implementation_commit",
        "implementation_tree",
        "successor_base_commit",
        "shared_path_list_sha256",
        "module_count",
        "modules",
        "repository_runtime_manifest_sha256",
    }
)
_REPOSITORY_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "module_name",
        "relative_path",
        "role",
        "owning_revision",
        "git_blob_sha1",
        "literal_sha256",
        "byte_count",
    }
)
_EXECUTION_DEPENDENCY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "python_runtime",
        "git_runtime",
        "builtin_or_frozen_modules",
        "module_files",
        "distributions",
        "loaded_binaries",
        "timezone_files",
        "normalized_pseudo_module_aliases",
        "sys_path_sha256",
        "counts",
        "execution_dependency_manifest_sha256",
    }
)
_PYTHON_RUNTIME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "executable_basename",
        "executable_path_sha256",
        "executable_bytes",
        "executable_sha256",
        "python_dll_basename",
        "python_dll_path_sha256",
        "python_dll_bytes",
        "python_dll_sha256",
        "python_version",
        "cache_tag",
        "os_name",
        "sys_platform",
        "launcher_profile_sha256",
        "bootstrap_sha256",
        "flags_sha256",
        "process_environment_sha256",
    }
)
_GIT_RUNTIME_FIELDS: Final[frozenset[str]] = frozenset(
    {"basename", "resolved_path_sha256", "byte_count", "literal_sha256", "version_output_sha256"}
)
_BUILTIN_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"module_name", "origin_kind"}
)
_MODULE_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "module_name",
        "origin_kind",
        "resolved_path_sha256",
        "member_name",
        "distribution_name",
        "byte_count",
        "literal_sha256",
    }
)
_DISTRIBUTION_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"normalized_name", "version", "record_sha256", "file_count", "ordered_files_sha256"}
)
_BINARY_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"basename", "resolved_path_sha256", "byte_count", "literal_sha256"}
)
_TIMEZONE_RUNTIME_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"zone_key", "provider", "resolved_path_sha256", "byte_count", "literal_sha256"}
)
_PSEUDO_MODULE_ALIAS_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "alias_name",
        "owner_module",
        "owner_attribute",
        "object_kind",
        "object_type",
        "metadata_sha256",
        "owner_origin_sha256",
        "row_sha256",
    }
)
_LOADED_CODE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "checkpoint_name",
        "repository_runtime_manifest_sha256",
        "execution_dependency_manifest_sha256",
        "module_count",
        "module_rows",
        "callable_count",
        "callable_rows",
        "loaded_code_manifest_sha256",
    }
)
_LOADED_MODULE_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"module_name", "origin_kind", "origin_identity_sha256", "namespace_sha256"}
)
_LOADED_CALLABLE_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "qualified_name",
        "kind",
        "owner_module",
        "owner_file_sha256",
        "object_type",
        "code_sha256",
        "defaults_sha256",
        "kwdefaults_sha256",
        "annotations_sha256",
        "closure_sha256",
        "referenced_globals_sha256",
        "descriptor_members_sha256",
        "owning_binary_sha256",
        "owner_slots_sha256",
    }
)

_BRIDGE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "document_count",
        "filename_present_count",
        "filename_missing_count",
        "source_authority_pins_sha256",
        "source_authority_base_commit",
        "source_authority_base_tree",
        "source_inventory_sha256",
        "stage_source_seal_sha256",
        "checkpoint_sha256",
        "compact_replay_sha256",
        "role_manifests_sha256",
        "role_plan_sha256",
        "science_projection_sha256",
        "legacy_projection_sha256",
        "compatibility_manifest_sha256",
        "universe_sha256",
        "content_manifest_sha256",
        "calendar_sessions_sha256",
        "universe_event_proofs_sha256",
        "documents_sha256",
        "records_sha256",
        "source_order_sha256",
        "event_order_sha256",
        "prior_links_sha256",
        "primary_documents_sha256",
        "set_parity",
        "source_sequence_parity",
        "legacy_projection_parity",
        "typed_identity_parity",
        "nullable_filename_parity",
        "no_fabricated_primary_url",
        "prior_links_are_internal_only",
        "first_10k_and_10q_have_no_prior",
        "peak_live_complete_submission_blob_count",
        "sec_request_count",
        "confirmation_or_final_opened",
        "contains_private_rows",
        "contains_accessions_urls_filenames_or_bodies",
        "bridge_sha256",
    }
)
_REQUEST_COMMITMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "document_count",
        "record_count",
        "event_count",
        "request_count",
        "pilot_count",
        "remaining_count",
        "filename_present_count",
        "filename_missing_count",
        "documents_sha256",
        "records_sha256",
        "events_sha256",
        "compatibility_manifest_sha256",
        "universe_sha256",
        "content_manifest_sha256",
        "calendar_sessions_sha256",
        "universe_event_proofs_sha256",
        "preprocessed_events_sha256",
        "canonical_requests_sha256",
        "model_slice_sha256",
        "canonical_request_index_sha256",
        "model_plan_sha256",
        "pilot_order_sha256",
        "remaining_order_sha256",
        "minimum_request_byte_count",
        "maximum_request_byte_count",
        "source_order_sha256",
        "prior_links_sha256",
        "set_parity",
        "sequence_parity",
        "prior_link_parity",
        "confirmation_or_final_opened",
        "contains_private_rows",
    }
)
_SOURCE_INVENTORY_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"path", "git_blob_sha1", "literal_sha256", "byte_count"}
)
_REPOSITORY_SNAPSHOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ancestry",
        "source_inventory",
        "source_inventory_sha256",
        "production_source_inventory_sha256",
        "test_source_inventory_sha256",
        "predecessor_inventory_sha256",
        "unchanged_predecessor_inventory_sha256",
        "local_production_closure_manifest_sha256",
        "local_production_closure_paths_sha256",
        "local_import_paths",
        "local_import_paths_sha256",
        "package_initializer_predecessor_equal",
        "preregistration_authenticated",
    }
)
_QUALIFICATION_REPORT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "status",
        "phase_count",
        "deadline_seconds",
        "durability_mode",
        "command_profile_sha256",
        "phases",
        "private_aggregate_sha256",
        "qualification_sha256",
    }
)
_QUALIFICATION_PUBLIC_PHASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "node_count",
        "node_list_sha256",
        "collection_duration_ns",
        "execution_duration_ns",
        "collection_exit_code",
        "execution_exit_code",
        "passed_count",
        "failed_count",
        "error_count",
        "skipped_count",
        "xfailed_count",
        "xpassed_count",
        "collection_result_sha256",
        "execution_result_sha256",
        "collection_log_sha256",
        "execution_log_sha256",
        "collection_xml_sha256",
        "execution_xml_sha256",
        "collection_manifest_sha256",
    }
)
_QUALIFICATION_INTENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "phase",
        "mode",
        "argv",
        "environment_profile",
        "repository_commit",
        "repository_tree",
        "clean_state_sha256",
        "deadline_monotonic_ns",
        "runtime",
        "collection_manifest_sha256",
        "intent_sha256",
    }
)
_QUALIFICATION_RUNTIME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "python_version",
        "python_cache_tag",
        "os_name",
        "sys_platform",
        "executable_path",
        "executable_basename",
        "executable_bytes",
        "executable_sha256",
        "pytest_version",
        "pytest_init_path",
        "pytest_init_bytes",
        "pytest_init_sha256",
    }
)
_QUALIFICATION_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "passed_count",
    "failed_count",
    "error_count",
    "skipped_count",
    "xfailed_count",
    "xpassed_count",
)
_QUALIFICATION_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "phase",
        "mode",
        "status",
        "intent_sha256",
        "collection_manifest_sha256",
        "started_unix_ns",
        "ended_unix_ns",
        "duration_monotonic_ns",
        "node_count",
        "node_list_sha256",
        "node_ids",
        *_QUALIFICATION_COUNT_FIELDS,
        "exit_code",
        "timed_out",
        "deadline_overrun",
        "terminal_reason",
        "exception_code",
        "log_bytes",
        "log_sha256",
        "stdout_bytes",
        "stdout_sha256",
        "xml_present",
        "xml_bytes",
        "xml_sha256",
        "result_sha256",
    }
)
_QUALIFICATION_COLLECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "phase",
        "status",
        "ordered_node_ids",
        "node_count",
        "node_list_sha256",
        "collection_result_sha256",
        "collection_completion_sha256",
        "collection_manifest_sha256",
    }
)
_QUALIFICATION_COMPLETION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "phase",
        "mode",
        "status",
        "terminal_reason",
        "deadline_overrun",
        "result_sha256",
        "result_literal_sha256",
        "log_sha256",
        "stdout_sha256",
        "xml_sha256",
        "completion_sha256",
    }
)
_QUALIFICATION_PRIVATE_PHASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "node_count",
        "node_list_sha256",
        "collection_manifest_sha256",
        "collection_completion_sha256",
        "execution_completion_sha256",
        "collection",
        "execution",
    }
)
_QUALIFICATION_PRIVATE_AGGREGATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "suite_identity",
        "status",
        "deadline_monotonic_ns",
        "deadline_seconds",
        "durability_mode",
        "repository_commit",
        "repository_tree",
        "clean_state_sha256",
        "runtime",
        "phases",
        "private_aggregate_sha256",
    }
)
_PRIVATE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "contract_manifest_sha256",
        "repository",
        "bridge",
        "request_commitments",
        "effect_counts_before",
        "effect_counts_after",
        "qualification",
        "runtime_binding_authority_sha256",
        "privacy",
        "gates",
        "private_manifest_sha256",
    }
)
_PUBLIC_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "stage",
        "branch",
        "preregistration_commit",
        "preregistration_tree",
        "implementation_commit",
        "implementation_tree",
        "contract_manifest_sha256",
        "science_projection_sha256",
        "source_authority",
        "counts",
        "aggregates",
        "effect_counts",
        "qualification",
        "runtime_binding_authority_sha256",
        "privacy",
        "gates",
        "private_manifest_sha256",
        "private_manifest_literal_sha256",
        "eligible_for_public_seal",
        "development_authorized",
        "public_artifact_sha256",
    }
)
_PUBLIC_FAILED_PREFLIGHT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "stage",
        "branch",
        "failure_code",
        "preflight_consumed",
        "failure_preserved",
        "redacted_error_only",
        "eligible_for_public_seal",
        "rerun_authorized",
        "development_authorized",
        "confirmation_and_final_opened",
        "real_money_authorized",
        "public_artifact_sha256",
    }
)
_PUSHED_RESULT_PAUSE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "status",
        "stage",
        "attempt_id",
        "branch",
        "preflight_commit",
        "implementation_commit",
        "source_bridge_sha256",
        "science_projection_sha256",
        "model_plan_sha256",
        "pilot_count",
        "pilot_durations_ns",
        "formula",
        "projected_ns",
        "threshold_ns",
        "strictly_greater_pause",
        "pilot_guard_sha256",
        "latency_receipt_sha256",
        "remaining_order_sha256",
        "effect_report",
        "model_responses_opened",
        "market_values_opened",
        "sixth_generation_attempted",
        "continuation_requires_fresh_explicit_permission",
        "confirmation_and_final_opened",
        "privacy_passed",
        "pause_artifact_sha256",
    }
)
_PUSHED_RESULT_PILOT_GUARD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "segment_id",
        "store_segment_id",
        "generation_count",
        "pre_runtime_receipt_sha256",
        "post_runtime_receipt_sha256",
        "stable_runtime_identity_sha256",
        "ordered_generation_response_event_sha256s",
        "ordered_generation_response_events_sha256",
        "raw_show_hash_is_diagnostic_only",
        "modified_at_is_excluded_only",
        "identity_http_request_count",
        "retry_count",
        "segment_guard_sha256",
    }
)
_PUSHED_RESULT_LATENCY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "selection",
        "remaining_order",
        "pilot_rows",
        "pilot_order_sha256",
        "pilot_count",
        "remaining_count",
        "formula",
        "projected_ns",
        "threshold_ns",
        "pause_required",
        "timing_interval",
        "latency_receipt_sha256",
    }
)
_PUSHED_RESULT_LATENCY_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"execution_ordinal", "request_sha256", "request_byte_count", "duration_ns"}
)
_PRIVATE_GATES: Final[frozenset[str]] = frozenset(
    {
        "implementation_ancestry_passed",
        "exact_twelve_file_delta_passed",
        "predecessor_blobs_unchanged",
        "source_authority_authenticated",
        "streaming_projection_parity_passed",
        "request_and_pilot_commitments_frozen",
        "confirmation_and_final_unreachable",
        "zero_external_effects",
        "qualification_passed",
        "runtime_authority_created",
        "privacy_passed",
        "eligible_for_public_seal",
    }
)
_PUBLIC_GATES: Final[frozenset[str]] = _PRIVATE_GATES - {
    "eligible_for_public_seal"
}
_PRIVATE_PRIVACY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "aggregate_only_bridge_serialized",
        "aggregate_only_request_commitments_serialized",
        "readable_contact_stored",
        "compact_manifest_absolute_private_path_stored",
        "qualification_intent_absolute_paths_stored",
    }
)
_PUBLIC_PRIVACY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "readable_contact_published",
        "accessions_urls_filenames_offsets_bodies_published",
        "private_paths_published",
        "canonical_requests_published",
        "pre_release_science_values_published",
        "redacted_errors_only",
    }
)


class V315PreflightError(RuntimeError):
    """A fixed public-safe preflight rejection."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or re.fullmatch(r"[a-z0-9_]{3,80}", code) is None:
            code = "preflight_rejected"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PreflightDependencies:
    """Injected read-only/pure dependencies for the once-only preflight."""

    inspect_repository: Callable[[Path], Mapping[str, Any]]
    create_runtime_authority: Callable[
        [Path, Mapping[str, Any]], Mapping[str, Any]
    ]
    authenticate_source: Callable[[Path], Any]
    build_projection: Callable[[Any], Any]
    build_request_commitments: Callable[[Any], Mapping[str, Any]]
    effect_snapshot: Callable[[], Mapping[str, Any]]
    run_qualification: Callable[[Path], Mapping[str, Any]]
    privacy_tokens: Callable[[Path], Sequence[bytes | str]]


@dataclass(frozen=True)
class PushedResultDependencies:
    """Injected read-only dependencies for the separately pushed result gate."""

    authenticate_preflight_revision: Callable[[Path, str], Mapping[str, Any]]
    load_private_contact: Callable[[Path], str]
    authenticate_source: Callable[[Path, str], Any]
    build_projection: Callable[[Any], Any]
    rebuild_attempt_context: Callable[
        [Mapping[str, Any], Any], Mapping[str, Any]
    ]
    open_store: Callable[[Path, Mapping[str, Any]], Any]
    build_effect_report: Callable[[Any], Mapping[str, Any]]
    build_public_effect_report: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    rebuild_pilot_evidence: Callable[..., tuple[Mapping[str, Any], Mapping[str, Any]]]
    build_public_pause_artifact: Callable[..., Mapping[str, Any]]
    build_continuation_preregistration: Callable[[Mapping[str, Any]], bytes]
    replay_completed_terminal: Callable[..., Mapping[str, Any]]
    replay_failure_terminal: Callable[..., Mapping[str, Any]]
    build_public_terminal_artifact: Callable[..., Mapping[str, Any]]
    build_comparison_update: Callable[[bytes, Mapping[str, Any]], bytes]
    privacy_tokens: Callable[[Path, str], Sequence[bytes | str]]


@dataclass(frozen=True)
class PublicationRecoveryDependencies:
    """Read-only builders used to finish an interrupted public publication."""

    authenticate_preflight_revision: Callable[[Path, str], Mapping[str, Any]]
    open_store: Callable[[Path, Mapping[str, Any]], Any]
    build_public_terminal_artifact: Callable[..., Mapping[str, Any]]
    build_comparison_update: Callable[[bytes, Mapping[str, Any]], bytes]
    rebuild_public_pause_artifact: Callable[..., Mapping[str, Any]] | None = None


def _reject(code: str) -> None:
    raise V315PreflightError(code)


def _is_sha1(value: Any) -> bool:
    return type(value) is str and _SHA1_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _reject(code)
    return _plain_json_mapping(value, code=code)


def _plain_json_mapping(value: Any, *, code: str) -> dict[str, Any]:
    """Recursively thaw MappingProxy/tuple evidence through canonical JSON."""

    def thaw(item: Any, *, depth: int) -> Any:
        if depth > 200:
            _reject(code)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if type(key) is not str or key in result:
                    _reject(code)
                result[key] = thaw(child, depth=depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [thaw(child, depth=depth + 1) for child in item]
        if item is None or type(item) in {str, bool, int, float}:
            return item
        _reject(code)

    detached = thaw(value, depth=0)
    if type(detached) is not dict:
        _reject(code)
    try:
        contract.canonical_json_bytes(detached)
    except Exception:
        _reject(code)
    return detached


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return contract.canonical_json_bytes(dict(value)) + b"\n"


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _reject("preflight_self_hash_invalid")
    detached = copy.deepcopy(dict(value))
    detached[field] = contract.canonical_sha256(detached)
    return detached


def _validate_self_hash(value: Any, field: str, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _reject(code)
    detached = _plain_json_mapping(value, code=code)
    observed = detached.pop(field, None)
    if not _is_sha256(observed) or observed != contract.canonical_sha256(detached):
        _reject(code)
    return _plain_json_mapping(value, code=code)


def private_windows_path_sha256(path: str | os.PathLike[str]) -> str:
    """Return the preregistered private token for one existing Windows path."""

    try:
        resolved = str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError, ValueError, TypeError):
        _reject("runtime_path_invalid")
    normalized = ntpath.normcase(ntpath.normpath(resolved)).replace("/", "\\")
    drive, tail = ntpath.splitdrive(normalized)
    if normalized.endswith("\\") and tail not in {"\\", ""}:
        normalized = normalized.rstrip("\\")
    return hashlib.sha256(
        _PRIVATE_PATH_DOMAIN + b"\x00" + normalized.encode("utf-8")
    ).hexdigest()


def _resolve_authenticated_git_runtime() -> tuple[Path, bytes, bytes]:
    """Reproduce the outer launcher's read-only Git identity check."""

    candidate = shutil.which("git")
    if type(candidate) is not str or not candidate:
        _reject("creation_pretrust_git_invalid")
    try:
        discovered = Path(candidate).resolve(strict=True)
        alternatives = [discovered]
        if discovered.parent.name.casefold() == "cmd":
            alternatives.insert(0, discovered.parent.parent / "bin" / "git.exe")
        executable: Path | None = None
        for alternative in alternatives:
            try:
                resolved = alternative.resolve(strict=True)
                observed = resolved.lstat()
            except OSError:
                continue
            if (
                resolved.name.casefold() == "git.exe"
                and stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and not getattr(observed, "st_file_attributes", 0)
                & _REPARSE_ATTRIBUTE
                and observed.st_nlink == 1
            ):
                executable = resolved
                break
        if executable is None:
            _reject("creation_pretrust_git_invalid")
        details = executable.lstat()
        payload = _read_regular_file(
            executable, maximum=None, code="creation_pretrust_git_invalid"
        )
        completed = subprocess.run(
            [str(executable), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _reject("creation_pretrust_git_invalid")
    if (
        executable.name.casefold() != "git.exe"
        or not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or not payload
        or completed.returncode != 0
        or not completed.stdout.startswith(b"git version ")
    ):
        _reject("creation_pretrust_git_invalid")
    return executable, payload, completed.stdout


def validate_creation_pretrust_attestation(
    value: Any,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate, but never mint, the outer stdlib launcher's one-shot proof."""

    code = "creation_pretrust_attestation_invalid"
    item = _strict_mapping(value, _CREATION_PRETRUST_FIELDS, code=code)
    _validate_self_hash(item, "attestation_sha256", code=code)
    if type(item["source_inventory"]) is not list or len(
        item["source_inventory"]
    ) != len(contract.IMPLEMENTATION_ALLOWED_PATHS):
        _reject(code)
    rows: list[dict[str, Any]] = []
    for expected_path, candidate in zip(
        contract.IMPLEMENTATION_ALLOWED_PATHS,
        item["source_inventory"],
        strict=True,
    ):
        row = _strict_mapping(
            candidate,
            _CREATION_PRETRUST_SOURCE_ROW_FIELDS,
            code=code,
        )
        if (
            row["path"] != expected_path
            or not _is_sha1(row["git_blob_sha1"])
            or not _is_sha256(row["git_literal_sha256"])
            or not _is_sha256(row["working_literal_sha256"])
            or type(row["git_byte_count"]) is not int
            or row["git_byte_count"] <= 0
            or type(row["working_byte_count"]) is not int
            or row["working_byte_count"] <= 0
            or row["git_literal_sha256"] != row["working_literal_sha256"]
            or row["git_byte_count"] != row["working_byte_count"]
        ):
            _reject(code)
        rows.append(row)
    git_runtime = _strict_mapping(
        item["git_runtime"], _CREATION_PRETRUST_GIT_FIELDS, code=code
    )
    python_runtime = _strict_mapping(
        item["python_runtime"], _CREATION_PRETRUST_PYTHON_FIELDS, code=code
    )
    bootstrap = _strict_mapping(
        item["bootstrap"], _CREATION_PRETRUST_BOOTSTRAP_FIELDS, code=code
    )
    try:
        effects = contract.validate_effect_counts(
            item["effect_counts"], route="zero_effect_preflight"
        )
        executable = Path(sys.executable).resolve(strict=True)
        executable_payload = _read_regular_file(
            executable, maximum=None, code=code
        )
        git_executable, git_payload, git_version = (
            _resolve_authenticated_git_runtime()
        )
    except V315PreflightError:
        raise
    except Exception:
        _reject(code)
    if (
        item["schema_version"]
        != CREATION_PRETRUST_ATTESTATION_SCHEMA_VERSION
        or item["base_commit"] != contract.BASE_COMMIT
        or item["base_tree"] != contract.BASE_TREE
        or item["preregistration_commit"] != contract.PREREGISTRATION_COMMIT
        or item["preregistration_tree"] != contract.PREREGISTRATION_TREE
        or item["preregistration_git_blob_sha1"]
        != contract.PREREGISTRATION_GIT_BLOB_SHA1
        or item["preregistration_literal_sha256"]
        != contract.PREREGISTRATION_LITERAL_SHA256
        or item["preregistration_literal_bytes"]
        != contract.PREREGISTRATION_LITERAL_BYTES
        or not _is_sha1(item["implementation_commit"])
        or not _is_sha1(item["implementation_tree"])
        or item["implementation_parent"] != contract.PREREGISTRATION_COMMIT
        or item["branch"] != contract.BRANCH_NAME
        or item["upstream"] != f"origin/{contract.BRANCH_NAME}"
        or item["origin_url"] != CREATION_PRETRUST_EXPECTED_ORIGIN_URL
        or item["repository_root_sha256"]
        != private_windows_path_sha256(repository_root)
        or item["source_inventory_sha256"]
        != contract.canonical_sha256(rows)
        or item["clean_worktree"] is not True
        or git_runtime["basename"].casefold() != "git.exe"
        or git_runtime["basename"] != git_executable.name
        or git_runtime["resolved_path_sha256"]
        != private_windows_path_sha256(git_executable)
        or git_runtime["byte_count"] != len(git_payload)
        or git_runtime["literal_sha256"]
        != hashlib.sha256(git_payload).hexdigest()
        or git_runtime["version_output_sha256"]
        != hashlib.sha256(git_version).hexdigest()
        or python_runtime["basename"]
        != contract.QUALIFICATION_EXECUTABLE_BASENAME
        or python_runtime["basename"] != executable.name
        or python_runtime["resolved_path_sha256"]
        != private_windows_path_sha256(executable)
        or python_runtime["byte_count"] != len(executable_payload)
        or python_runtime["literal_sha256"]
        != hashlib.sha256(executable_payload).hexdigest()
        or python_runtime["python_version"]
        != contract.QUALIFICATION_PYTHON_VERSION
        or python_runtime["cache_tag"]
        != contract.QUALIFICATION_PYTHON_CACHE_TAG
        or type(bootstrap["byte_count"]) is not int
        or bootstrap["byte_count"] != len(SCIENTIFIC_BOOTSTRAP_BYTES)
        or bootstrap["literal_sha256"] != SCIENTIFIC_BOOTSTRAP_SHA256
    ):
        _reject(code)
    # Keep the already resolved identities process-locally.  Qualification and
    # runtime derivation must reuse these exact objects/bytes; resolving a
    # second Git from a caller-controlled PATH would create a new trust root.
    active_context = _runtime_active_preflight_context()
    active_context["trusted_git"] = (
        git_executable,
        git_payload,
        git_version,
    )
    active_context["trusted_python"] = (
        executable,
        executable_payload,
    )
    return {
        **item,
        "source_inventory": rows,
        "git_runtime": git_runtime,
        "python_runtime": python_runtime,
        "bootstrap": bootstrap,
        "effect_counts": effects,
    }


def _consume_creation_pretrust_attestation(root: Path) -> dict[str, Any]:
    """Consume the process-local outer proof before any private mutation."""

    if not hasattr(builtins, CREATION_PRETRUST_SENTINEL_NAME):
        _reject("creation_pretrust_attestation_missing")
    try:
        value = getattr(builtins, CREATION_PRETRUST_SENTINEL_NAME)
        delattr(builtins, CREATION_PRETRUST_SENTINEL_NAME)
    except Exception:
        _reject("creation_pretrust_attestation_invalid")
    validated = validate_creation_pretrust_attestation(value, repository_root=root)
    _runtime_active_preflight_context()["repository_root_sha256"] = (
        private_windows_path_sha256(root)
    )
    return validated


def _validate_creation_pretrust_repository_binding(
    attestation: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> None:
    ancestry = repository["ancestry"]
    observed_rows = repository["source_inventory"]
    attested_rows = attestation["source_inventory"]
    if (
        ancestry["commit"] != attestation["implementation_commit"]
        or ancestry["tree"] != attestation["implementation_tree"]
        or ancestry["parent"] != attestation["implementation_parent"]
        or ancestry["branch"] != attestation["branch"]
        or ancestry["clean_worktree"] is not True
        or len(observed_rows) != len(attested_rows)
    ):
        _reject("creation_pretrust_repository_mismatch")
    for observed, attested in zip(observed_rows, attested_rows, strict=True):
        if (
            observed["path"] != attested["path"]
            or observed["git_blob_sha1"] != attested["git_blob_sha1"]
            or observed["literal_sha256"]
            != attested["git_literal_sha256"]
            or observed["literal_sha256"]
            != attested["working_literal_sha256"]
            or observed["byte_count"] != attested["git_byte_count"]
            or observed["byte_count"] != attested["working_byte_count"]
        ):
            _reject("creation_pretrust_repository_mismatch")


def derive_final_sys_path(
    initial_entries: Sequence[str],
    *,
    python_executable: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    purelib_root: str | os.PathLike[str],
    platlib_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate Python 3.12.2 ``-S`` search roots and derive one sealed path."""

    if (
        not isinstance(initial_entries, Sequence)
        or isinstance(initial_entries, (str, bytes, bytearray))
        or len(initial_entries) != 5
        or any(type(item) is not str for item in initial_entries)
        or initial_entries[0] != ""
    ):
        _reject("runtime_initial_sys_path_invalid")
    try:
        executable = Path(python_executable).resolve(strict=True)
        parent = executable.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        _reject("runtime_initial_sys_path_invalid")
    lexical_zip = parent / "python312.zip"
    expected_lexical = ntpath.normcase(ntpath.normpath(str(lexical_zip)))
    observed_lexical = ntpath.normcase(ntpath.normpath(initial_entries[1]))
    if observed_lexical != expected_lexical or lexical_zip.exists():
        _reject("runtime_initial_sys_path_invalid")
    expected_existing = (parent / "DLLs", parent / "Lib", parent)
    observed_existing = initial_entries[2:]
    for observed, expected in zip(observed_existing, expected_existing, strict=True):
        try:
            if Path(observed).resolve(strict=True) != expected.resolve(strict=True):
                _reject("runtime_initial_sys_path_invalid")
        except (OSError, RuntimeError, ValueError):
            _reject("runtime_initial_sys_path_invalid")

    ordered = [Path(repository_root), *(Path(item) for item in observed_existing)]
    ordered.extend((Path(purelib_root), Path(platlib_root)))
    resolved_paths: list[Path] = []
    tokens: list[str] = []
    seen: set[str] = set()
    for entry in ordered:
        try:
            resolved = entry.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            _reject("runtime_final_sys_path_invalid")
        token = private_windows_path_sha256(resolved)
        if token in seen:
            continue
        seen.add(token)
        resolved_paths.append(resolved)
        tokens.append(token)
    return {
        "resolved_paths": tuple(resolved_paths),
        "final_sys_path_tokens": tokens,
        "final_sys_path_sha256": contract.canonical_sha256(tokens),
    }


def build_launcher_profile(
    *,
    python_executable_sha256: str,
    bootstrap_bytes: bytes,
) -> dict[str, Any]:
    if type(bootstrap_bytes) is not bytes or not bootstrap_bytes:
        _reject("runtime_bootstrap_invalid")
    body = {
        "schema_version": LAUNCHER_PROFILE_SCHEMA_VERSION,
        "python_executable_sha256": _sha256_value(
            python_executable_sha256, "runtime_launcher_profile_invalid"
        ),
        "python_flags": ["-s", "-S", "-B"],
        "bootstrap_sha256": hashlib.sha256(bootstrap_bytes).hexdigest(),
        "python_no_user_site": "1",
        "python_hash_seed": "0",
        "python_path": "",
    }
    return {**body, "launcher_profile_sha256": contract.canonical_sha256(body)}


def _sha256_value(value: Any, code: str) -> str:
    if not _is_sha256(value):
        _reject(code)
    return value


def build_reduced_runtime_environment(
    source: Mapping[str, str],
    *,
    git_executable: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the exact child environment and its path-redacted hash material."""

    if not isinstance(source, Mapping):
        _reject("runtime_environment_invalid")
    child: dict[str, str] = {}
    hash_material: dict[str, Any] = {}
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        value = source.get(name)
        if type(value) is not str or not value:
            _reject("runtime_environment_invalid")
        child[name] = value
        hash_material[name] = private_windows_path_sha256(value)
    path_directories = (
        Path(git_executable).resolve(strict=True).parent,
        Path(python_executable).resolve(strict=True).parent,
        Path(child["SYSTEMROOT"]) / "System32",
        Path(child["SYSTEMROOT"]),
    )
    raw_path: list[str] = []
    path_tokens: list[str] = []
    seen: set[str] = set()
    for directory in path_directories:
        token = private_windows_path_sha256(directory)
        if token in seen:
            continue
        seen.add(token)
        raw_path.append(str(directory.resolve(strict=True)))
        path_tokens.append(token)
    child["PATH"] = ";".join(raw_path)
    hash_material["PATH"] = path_tokens
    for name, value in _FIXED_RUNTIME_ENVIRONMENT.items():
        child[name] = value
        hash_material[name] = value
    if set(child) != set(RUNTIME_ENVIRONMENT_ALLOWLIST):
        _reject("runtime_environment_invalid")
    ordered_hash_material = {
        name: hash_material[name] for name in RUNTIME_ENVIRONMENT_ALLOWLIST
    }
    return {
        "environment_allowlist": list(RUNTIME_ENVIRONMENT_ALLOWLIST),
        "child_environment": child,
        "process_environment_material": ordered_hash_material,
        "process_environment_sha256": contract.canonical_sha256(
            ordered_hash_material
        ),
    }


def _git_blob_sha1_bytes(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def build_repository_runtime_manifest(
    repo_root: Path,
    *,
    implementation_commit: str,
    implementation_tree: str,
    origin_url: str = "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
) -> dict[str, Any]:
    """Build and Git-authenticate the exact 44-row repository authority."""

    if not _is_sha1(implementation_commit) or not _is_sha1(implementation_tree):
        _reject("repository_runtime_identity_invalid")
    paths = [
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
        *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    ]
    if len(paths) != 44 or len(paths) != len(set(paths)):
        _reject("repository_runtime_path_set_invalid")
    rows: list[dict[str, Any]] = []
    for relative_path in sorted(paths, key=lambda value: value.encode("utf-8")):
        target = repo_root / Path(relative_path)
        try:
            details = target.lstat()
            resolved = target.resolve(strict=True)
            resolved.relative_to(repo_root.resolve(strict=True))
            payload = _read_regular_file(
                target, maximum=None, code="repository_runtime_file_invalid"
            )
        except (OSError, ValueError):
            _reject("repository_runtime_file_invalid")
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            or details.st_nlink not in {0, 1}
            or not payload
        ):
            _reject("repository_runtime_file_invalid")
        if relative_path.endswith("/__init__.py"):
            module_name = relative_path[: -len("/__init__.py")].replace("/", ".")
        else:
            module_name = relative_path[:-3].replace("/", ".")
        if relative_path in contract.IMPLEMENTATION_PRODUCTION_PATHS:
            role = "v315"
            owner = implementation_commit
        elif relative_path == contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH:
            role = "package_bootstrap"
            owner = _RUNTIME_SUCCESSOR_BASE_COMMIT
        else:
            role = "shared"
            owner = _RUNTIME_SUCCESSOR_BASE_COMMIT
        committed = _git(repo_root, "show", f"{owner}:{relative_path}", binary=True)
        committed_blob = _git(repo_root, "rev-parse", f"{owner}:{relative_path}")
        implementation_payload = _git(
            repo_root,
            "show",
            f"{implementation_commit}:{relative_path}",
            binary=True,
        )
        implementation_blob = _git(
            repo_root,
            "rev-parse",
            f"{implementation_commit}:{relative_path}",
        )
        if (
            type(committed) is not bytes
            or type(implementation_payload) is not bytes
            or type(committed_blob) is not str
            or type(implementation_blob) is not str
            or committed != payload
            or implementation_payload != payload
            or committed_blob != implementation_blob
            or committed_blob != _git_blob_sha1_bytes(payload)
        ):
            _reject("repository_runtime_git_binding_invalid")
        rows.append(
            {
                "module_name": module_name,
                "relative_path": relative_path,
                "role": role,
                "owning_revision": owner,
                "git_blob_sha1": committed_blob,
                "literal_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
        )
    body = {
        "schema_version": REPOSITORY_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "branch": contract.BRANCH_NAME,
        "origin_url": origin_url,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "successor_base_commit": _RUNTIME_SUCCESSOR_BASE_COMMIT,
        "shared_path_list_sha256": contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256,
        "module_count": len(rows),
        "modules": rows,
    }
    return {
        **body,
        "repository_runtime_manifest_sha256": contract.canonical_sha256(body),
    }


def validate_repository_runtime_manifest(value: Any) -> dict[str, Any]:
    code = "repository_runtime_manifest_invalid"
    item = _strict_mapping(value, _REPOSITORY_RUNTIME_FIELDS, code=code)
    _validate_self_hash(item, "repository_runtime_manifest_sha256", code=code)
    paths = [
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
        *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    ]
    expected_paths = sorted(paths, key=lambda path: path.encode("utf-8"))
    if (
        item["schema_version"] != REPOSITORY_RUNTIME_MANIFEST_SCHEMA_VERSION
        or item["branch"] != contract.BRANCH_NAME
        or item["origin_url"] != CREATION_PRETRUST_EXPECTED_ORIGIN_URL
        or not _is_sha1(item["implementation_commit"])
        or not _is_sha1(item["implementation_tree"])
        or item["successor_base_commit"] != _RUNTIME_SUCCESSOR_BASE_COMMIT
        or item["shared_path_list_sha256"]
        != contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
        or item["module_count"] != 44
        or type(item["modules"]) is not list
        or len(item["modules"]) != 44
    ):
        _reject(code)
    rows: list[dict[str, Any]] = []
    for expected_path, raw in zip(expected_paths, item["modules"], strict=True):
        row = _strict_mapping(raw, _REPOSITORY_RUNTIME_ROW_FIELDS, code=code)
        expected_module = (
            expected_path[: -len("/__init__.py")].replace("/", ".")
            if expected_path.endswith("/__init__.py")
            else expected_path[:-3].replace("/", ".")
        )
        if expected_path in contract.IMPLEMENTATION_PRODUCTION_PATHS:
            expected_role = "v315"
            expected_owner = item["implementation_commit"]
        elif expected_path == contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH:
            expected_role = "package_bootstrap"
            expected_owner = _RUNTIME_SUCCESSOR_BASE_COMMIT
        else:
            expected_role = "shared"
            expected_owner = _RUNTIME_SUCCESSOR_BASE_COMMIT
        if (
            row["relative_path"] != expected_path
            or row["module_name"] != expected_module
            or row["role"] != expected_role
            or row["owning_revision"] != expected_owner
            or not _is_sha1(row["git_blob_sha1"])
            or not _is_sha256(row["literal_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
        ):
            _reject(code)
        rows.append(row)
    return {**item, "modules": rows}


def _runtime_python_flags() -> tuple[dict[str, Any], str]:
    names = (
        "debug",
        "inspect",
        "interactive",
        "optimize",
        "dont_write_bytecode",
        "no_user_site",
        "no_site",
        "ignore_environment",
        "verbose",
        "bytes_warning",
        "quiet",
        "hash_randomization",
        "isolated",
        "dev_mode",
        "utf8_mode",
        "warn_default_encoding",
        "safe_path",
        "int_max_str_digits",
    )
    expected = (
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        1,
        0,
        False,
        4300,
    )
    values = tuple(getattr(sys.flags, name, None) for name in names)
    if values != expected or any(
        (type(value) is not type(wanted))
        for value, wanted in zip(values, expected, strict=True)
    ):
        _reject("runtime_python_flags_invalid")
    material = dict(zip(names, values, strict=True))
    return material, contract.canonical_sha256(material)


def _runtime_loaded_image_paths() -> tuple[Path, ...]:
    """Enumerate the current process image set without filesystem scanning."""

    if sys.platform != "win32":
        candidates = {Path(sys.executable).resolve(strict=True)}
        for module in tuple(sys.modules.values()):
            raw = getattr(module, "__file__", None)
            if type(raw) is str and raw.lower().endswith((".so", ".dylib")):
                try:
                    candidates.add(Path(raw).resolve(strict=True))
                except OSError:
                    _reject("runtime_loaded_binary_invalid")
        return tuple(sorted(candidates, key=lambda path: str(path).encode("utf-8")))
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = _wintypes.HANDLE
        psapi.EnumProcessModules.argtypes = [
            _wintypes.HANDLE,
            ctypes.POINTER(_wintypes.HMODULE),
            _wintypes.DWORD,
            ctypes.POINTER(_wintypes.DWORD),
        ]
        psapi.EnumProcessModules.restype = _wintypes.BOOL
        psapi.GetModuleFileNameExW.argtypes = [
            _wintypes.HANDLE,
            _wintypes.HMODULE,
            _wintypes.LPWSTR,
            _wintypes.DWORD,
        ]
        psapi.GetModuleFileNameExW.restype = _wintypes.DWORD
        process = kernel32.GetCurrentProcess()
        capacity = 256
        while True:
            array = (_wintypes.HMODULE * capacity)()
            needed = _wintypes.DWORD()
            if not psapi.EnumProcessModules(
                process,
                array,
                ctypes.sizeof(array),
                ctypes.byref(needed),
            ):
                raise OSError(ctypes.get_last_error(), "EnumProcessModules")
            count = int(needed.value) // ctypes.sizeof(_wintypes.HMODULE)
            if count <= capacity:
                break
            capacity = count + 32
        resolved: dict[str, Path] = {}
        for handle in array[:count]:
            size = 32768
            buffer = ctypes.create_unicode_buffer(size)
            length = int(
                psapi.GetModuleFileNameExW(process, handle, buffer, size)
            )
            if length <= 0 or length >= size:
                raise OSError(ctypes.get_last_error(), "GetModuleFileNameExW")
            path = Path(buffer.value).resolve(strict=True)
            token = private_windows_path_sha256(path)
            prior = resolved.get(token)
            if prior is not None and prior != path:
                _reject("runtime_loaded_binary_invalid")
            resolved[token] = path
        if not resolved:
            _reject("runtime_loaded_binary_invalid")
        return tuple(
            sorted(
                resolved.values(),
                key=lambda path: (
                    path.name.casefold().encode("utf-8"),
                    private_windows_path_sha256(path),
                ),
            )
        )
    except V315PreflightError:
        raise
    except Exception:
        _reject("runtime_loaded_binary_invalid")


def _runtime_binary_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        token = private_windows_path_sha256(path)
        if token in seen:
            _reject("runtime_loaded_binary_duplicate")
        seen.add(token)
        payload = _read_regular_file(
            path,
            maximum=None,
            code="runtime_loaded_binary_invalid",
            allow_hardlinks=True,
        )
        rows.append(
            {
                "basename": path.name,
                "resolved_path_sha256": token,
                "byte_count": len(payload),
                "literal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    rows.sort(
        key=lambda row: (
            row["basename"].casefold().encode("utf-8"),
            row["resolved_path_sha256"],
        )
    )
    return rows


def _runtime_timezone_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tzdata_version: str | None = None
    try:
        tzdata_version = importlib.metadata.version("tzdata")
    except importlib.metadata.PackageNotFoundError:
        pass
    for key in _RUNTIME_TIMEZONE_KEYS:
        # Constructing the object is part of preload and rejects an unavailable
        # or redirected scientific zone before any source/model body opens.
        try:
            observed = zoneinfo.ZoneInfo(key)
        except Exception:
            _reject("runtime_timezone_invalid")
        if observed.key != key:
            _reject("runtime_timezone_invalid")
        selected: Path | None = None
        for directory in zoneinfo.TZPATH:
            candidate = Path(directory).joinpath(*key.split("/"))
            if candidate.is_file():
                if selected is not None:
                    _reject("runtime_timezone_ambiguous")
                selected = candidate.resolve(strict=True)
        if selected is not None:
            payload = _read_regular_file(
                selected, maximum=4 * 1024 * 1024, code="runtime_timezone_invalid"
            )
            provider = "system_tzpath"
            token: str | None = private_windows_path_sha256(selected)
        else:
            if tzdata_version is None:
                _reject("runtime_timezone_invalid")
            try:
                resource = importlib.resources.files("tzdata.zoneinfo")
                for part in key.split("/"):
                    resource = resource.joinpath(part)
                with importlib.resources.as_file(resource) as resource_path:
                    payload = _read_regular_file(
                        Path(resource_path),
                        maximum=4 * 1024 * 1024,
                        code="runtime_timezone_invalid",
                    )
            except Exception:
                _reject("runtime_timezone_invalid")
            provider = f"tzdata:{tzdata_version}"
            token = None
        if not payload.startswith(b"TZif"):
            _reject("runtime_timezone_invalid")
        rows.append(
            {
                "zone_key": key,
                "provider": provider,
                "resolved_path_sha256": token,
                "byte_count": len(payload),
                "literal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    rows.sort(
        key=lambda row: (
            row["zone_key"].encode("utf-8"),
            row["provider"].encode("utf-8"),
            row["resolved_path_sha256"] or "",
        )
    )
    return rows


def _normalized_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if not normalized or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized) is None:
        _reject("runtime_distribution_name_invalid")
    return normalized


def _runtime_distribution_inventory(
    sealed_roots: Sequence[Path],
) -> tuple[dict[str, list[tuple[str, Any]]], dict[str, dict[str, Any]]]:
    """Parse every installed RECORD once and index exact owned files."""

    owners: dict[str, list[tuple[str, Any]]] = {}
    rows: dict[str, dict[str, Any]] = {}
    roots = tuple(path.resolve(strict=True) for path in sealed_roots)
    try:
        install_roots = tuple(
            dict.fromkeys(
                Path(value).resolve(strict=True)
                for value in (sys.base_prefix, sys.prefix)
            )
        )
    except (OSError, TypeError, ValueError):
        _reject("runtime_distribution_record_invalid")
    package_map = importlib.metadata.packages_distributions()
    required_names: set[str] = set()
    for module_name in sys.modules:
        required_names.update(package_map.get(module_name.split(".", 1)[0], ()))
    distributions: list[Any] = []
    for required_name in sorted(required_names, key=lambda value: value.encode("utf-8")):
        try:
            distributions.append(importlib.metadata.distribution(required_name))
        except importlib.metadata.PackageNotFoundError:
            _reject("runtime_distribution_invalid")
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        if type(raw_name) is not str or type(version) is not str or not version:
            _reject("runtime_distribution_invalid")
        name = _normalized_distribution_name(raw_name)
        key = f"{name}\x00{version}"
        files = distribution.files
        if files is None:
            continue
        record_candidates = [
            entry
            for entry in files
            if str(entry).replace("\\", "/").endswith(".dist-info/RECORD")
        ]
        if len(record_candidates) != 1:
            _reject("runtime_distribution_record_invalid")
        try:
            record_path = Path(
                distribution.locate_file(record_candidates[0])
            ).resolve(strict=True)
            record_payload = _read_regular_file(
                record_path,
                maximum=64 * 1024 * 1024,
                code="runtime_distribution_record_invalid",
            )
            decoded = record_payload.decode("utf-8", errors="strict")
            parsed = list(csv.reader(io.StringIO(decoded, newline="")))
        except (OSError, UnicodeError, csv.Error):
            _reject("runtime_distribution_record_invalid")
        ordered: list[dict[str, Any]] = []
        seen_files: set[str] = set()
        for record_row in parsed:
            if len(record_row) != 3 or not record_row[0]:
                _reject("runtime_distribution_record_invalid")
            record_name = record_row[0]
            if "\\" in record_name or record_name.startswith("/"):
                _reject("runtime_distribution_record_invalid")
            try:
                resolved = Path(distribution.locate_file(record_name)).resolve(
                    strict=True
                )
                if not any(
                    resolved.is_relative_to(root)
                    for root in (*roots, *install_roots)
                ):
                    _reject("runtime_distribution_record_invalid")
                payload = _read_regular_file(
                    resolved,
                    maximum=None,
                    code="runtime_distribution_record_invalid",
                )
            except (OSError, ValueError):
                _reject("runtime_distribution_record_invalid")
            if resolved.suffix.casefold() == ".pth":
                try:
                    lines = payload.decode("utf-8", errors="strict").splitlines()
                except UnicodeError:
                    _reject("runtime_distribution_record_invalid")
                if any(
                    line.lstrip().startswith(("import ", "import\t"))
                    for line in lines
                    if line.strip() and not line.lstrip().startswith("#")
                ):
                    _reject("runtime_distribution_executable_pth")
            token = private_windows_path_sha256(resolved)
            if token in seen_files:
                _reject("runtime_distribution_record_invalid")
            seen_files.add(token)
            ordered.append(
                {
                    "record_path": record_name,
                    "resolved_path_sha256": token,
                    "byte_count": len(payload),
                    "literal_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            owners.setdefault(token, []).append((key, distribution))
        row = {
            "normalized_name": name,
            "version": version,
            "record_sha256": hashlib.sha256(record_payload).hexdigest(),
            "file_count": len(ordered),
            "ordered_files_sha256": contract.canonical_sha256(ordered),
        }
        prior = rows.get(key)
        if prior is not None and prior != row:
            _reject("runtime_distribution_duplicate")
        rows[key] = row
    return owners, rows


def _runtime_repository_rows_by_module(
    repository: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    manifest = validate_repository_runtime_manifest(repository)
    return {row["module_name"]: row for row in manifest["modules"]}


def _runtime_validate_main_exclusion() -> None:
    module = sys.modules.get("__main__")
    frozen = sys.modules.get("_frozen_importlib")
    if (
        module is None
        or getattr(module, "__name__", None) != "__main__"
        or getattr(module, "__spec__", None) is not None
        or getattr(module, "__package__", None) is not None
        or hasattr(module, "__file__")
        or frozen is None
        or getattr(module, "__loader__", None)
        is not getattr(frozen, "BuiltinImporter", None)
    ):
        _reject("runtime_main_module_invalid")


def _runtime_module_inventory(
    *,
    repository_root: Path,
    repository_manifest: Mapping[str, Any],
    distribution_owners: Mapping[str, list[tuple[str, Any]]],
    distribution_rows: Mapping[str, dict[str, Any]],
    sealed_install_roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    repo_rows = _runtime_repository_rows_by_module(repository_manifest)
    repo_root = repository_root.resolve(strict=True)
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    platstdlib_root = Path(sysconfig.get_path("platstdlib")).resolve(strict=True)
    purelib_root = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    platlib_root = Path(sysconfig.get_path("platlib")).resolve(strict=True)
    builtins_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    used_distributions: dict[str, dict[str, Any]] = {}
    observed_repository: set[str] = set()
    _runtime_validate_main_exclusion()
    for module_name in sorted(sys.modules, key=lambda name: name.encode("utf-8")):
        if module_name == "__main__":
            continue
        module = sys.modules[module_name]
        if not isinstance(module, types.ModuleType):
            _reject("runtime_module_object_invalid")
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            builtins_rows.append(
                {
                    "module_name": module_name,
                    "origin_kind": "builtin" if origin == "built-in" else "frozen",
                }
            )
            continue
        raw_file = getattr(module, "__file__", None)
        if type(raw_file) is not str or not raw_file or type(origin) is not str:
            _reject("runtime_module_origin_invalid")

        repo_row = repo_rows.get(module_name)
        if repo_row is not None:
            expected = (repo_root / Path(repo_row["relative_path"])).resolve(
                strict=True
            )
            try:
                file_path = Path(raw_file).resolve(strict=True)
                origin_path = Path(origin).resolve(strict=True)
                details = expected.lstat()
            except OSError:
                _reject("runtime_repository_module_origin_invalid")
            if (
                file_path != expected
                or origin_path != expected
                or expected.suffix.casefold() != ".py"
                or not stat.S_ISREG(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
                or details.st_nlink not in {0, 1}
                or hashlib.sha256(
                    _read_regular_file(
                        expected,
                        maximum=None,
                        code="runtime_repository_module_origin_invalid",
                    )
                ).hexdigest()
                != repo_row["literal_sha256"]
            ):
                _reject("runtime_repository_module_origin_invalid")
            observed_repository.add(module_name)
            continue
        if module_name == "agent_benchmark" or module_name.startswith(
            "agent_benchmark."
        ):
            _reject("runtime_repository_module_not_allowed")

        lower_origin = origin.casefold()
        member_name: str | None = None
        if ".zip" in lower_origin:
            boundary = lower_origin.index(".zip") + 4
            archive = Path(origin[:boundary]).resolve(strict=True)
            member_name = origin[boundary:].lstrip("/\\").replace("\\", "/")
            if not member_name or ".." in member_name.split("/"):
                _reject("runtime_zip_member_invalid")
            try:
                with zipfile.ZipFile(archive, "r") as bundle:
                    payload = bundle.read(member_name)
            except (OSError, KeyError, zipfile.BadZipFile):
                _reject("runtime_zip_member_invalid")
            file_path = archive
            path_token = private_windows_path_sha256(archive)
            origin_kind = "zip_member"
        else:
            try:
                file_path = Path(origin).resolve(strict=True)
            except OSError:
                _reject("runtime_module_origin_invalid")
            payload = _read_regular_file(
                file_path, maximum=None, code="runtime_module_origin_invalid"
            )
            path_token = private_windows_path_sha256(file_path)
            suffix = file_path.suffix.casefold()
            in_stdlib = file_path.is_relative_to(stdlib_root) or file_path.is_relative_to(
                platstdlib_root
            )
            in_distribution = file_path.is_relative_to(
                purelib_root
            ) or file_path.is_relative_to(platlib_root)
            if suffix == ".pyd" or suffix in {".so", ".dylib"}:
                origin_kind = "extension_module"
            elif suffix == ".py":
                origin_kind = (
                    "distribution_source"
                    if in_distribution
                    else "stdlib_source" if in_stdlib else ""
                )
            elif suffix == ".pyc":
                origin_kind = (
                    "distribution_bytecode"
                    if in_distribution
                    else "stdlib_bytecode" if in_stdlib else ""
                )
            else:
                origin_kind = ""
            if not origin_kind:
                _reject("runtime_module_origin_invalid")

        is_distribution = origin_kind.startswith("distribution_") or (
            origin_kind == "extension_module"
            and (
                file_path.is_relative_to(purelib_root)
                or file_path.is_relative_to(platlib_root)
            )
        )
        distribution_name: str | None = None
        if is_distribution:
            owners = distribution_owners.get(path_token, [])
            if len(owners) != 1:
                _reject("runtime_module_distribution_owner_invalid")
            distribution_key, _distribution = owners[0]
            distribution = distribution_rows.get(distribution_key)
            if distribution is None:
                _reject("runtime_module_distribution_owner_invalid")
            distribution_name = distribution["normalized_name"]
            used_distributions[distribution_key] = distribution
        module_rows.append(
            {
                "module_name": module_name,
                "origin_kind": origin_kind,
                "resolved_path_sha256": path_token,
                "member_name": member_name,
                "distribution_name": distribution_name,
                "byte_count": len(payload),
                "literal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if observed_repository != set(repo_rows):
        _reject("runtime_repository_module_set_invalid")
    builtin_keys = [(row["module_name"], row["origin_kind"]) for row in builtins_rows]
    module_keys = [
        (
            row["module_name"],
            row["origin_kind"],
            row["resolved_path_sha256"],
            row["member_name"] or "",
        )
        for row in module_rows
    ]
    if len(builtin_keys) != len(set(builtin_keys)) or len(module_keys) != len(
        set(module_keys)
    ):
        _reject("runtime_module_duplicate")
    builtins_rows.sort(
        key=lambda row: (row["module_name"].encode("utf-8"), row["origin_kind"])
    )
    module_rows.sort(
        key=lambda row: (
            row["module_name"].encode("utf-8"),
            row["origin_kind"],
            row["resolved_path_sha256"],
            row["member_name"] or "",
        )
    )
    distributions = sorted(
        used_distributions.values(),
        key=lambda row: (
            row["normalized_name"].encode("utf-8"),
            row["version"].encode("utf-8"),
        ),
    )
    return builtins_rows, module_rows, distributions


def _runtime_pseudo_module_alias_seal(*, require_present: bool) -> dict[str, dict[str, Any]]:
    """Bind the four exact stdlib compatibility objects without broad aliases."""

    sealed: dict[str, dict[str, Any]] = {}
    for alias_name, owner_module, owner_attribute, object_kind, object_type in (
        _RUNTIME_PSEUDO_MODULE_ALIAS_SPECS
    ):
        owner = sys.modules.get(owner_module)
        if type(owner) is not types.ModuleType or not hasattr(owner, owner_attribute):
            _reject("runtime_pseudo_module_alias_invalid")
        value = getattr(owner, owner_attribute)
        if require_present:
            if alias_name not in sys.modules or sys.modules[alias_name] is not value:
                _reject("runtime_pseudo_module_alias_invalid")
        elif alias_name in sys.modules:
            _reject("runtime_pseudo_module_alias_invalid")
        observed_type = f"{type(value).__module__}.{type(value).__qualname__}"
        if (
            observed_type != object_type
            or (object_kind == "module" and type(value) is not types.ModuleType)
            or (
                object_kind == "deprecated_type"
                and (
                    isinstance(value, types.ModuleType)
                    or type(value).__module__ != "typing"
                    or type(value).__qualname__ != "_DeprecatedType"
                )
            )
            or getattr(value, "__name__", None) != alias_name
            or getattr(value, "__spec__", None) is not None
            or getattr(value, "__package__", None) is not None
            or getattr(value, "__loader__", None) is not None
            or hasattr(value, "__file__")
        ):
            _reject("runtime_pseudo_module_alias_invalid")
        metadata = {
            "alias_name": alias_name,
            "object_name": alias_name,
            "spec_is_null": True,
            "package_is_null": True,
            "loader_is_null": True,
            "file_attribute_present": False,
        }
        sealed[alias_name] = {
            "alias_name": alias_name,
            "owner_module": owner_module,
            "owner_attribute": owner_attribute,
            "object_kind": object_kind,
            "object_type": object_type,
            "metadata_sha256": contract.canonical_sha256(metadata),
            "object": value,
        }
    return sealed


def _runtime_normalize_pseudo_module_aliases() -> dict[str, dict[str, Any]]:
    """Require all four aliases once, seal them, then remove them once."""

    process_state = _runtime_process_state()
    if process_state["pseudo_module_normalization"] is not None:
        _reject("runtime_pseudo_module_alias_invalid")
    sealed = _runtime_pseudo_module_alias_seal(require_present=True)
    for alias_name in sorted(sealed, key=lambda value: value.encode("utf-8")):
        if sys.modules.pop(alias_name) is not sealed[alias_name]["object"]:
            _reject("runtime_pseudo_module_alias_invalid")
    _runtime_pseudo_module_alias_seal(require_present=False)
    process_state["pseudo_module_normalization"] = {
        alias_name: sealed[alias_name]["object"]
        for alias_name in sorted(sealed, key=lambda value: value.encode("utf-8"))
    }
    return sealed


def _runtime_pseudo_module_alias_rows(
    sealed: Mapping[str, Mapping[str, Any]],
    *,
    owner_origins: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Finish the durable four rows after the ordinary origin rows exist."""

    current = _runtime_pseudo_module_alias_seal(require_present=False)
    marker = _runtime_process_state()["pseudo_module_normalization"]
    expected_names = {
        spec[0] for spec in _RUNTIME_PSEUDO_MODULE_ALIAS_SPECS
    }
    if (
        set(sealed) != expected_names
        or set(current) != expected_names
        or type(marker) is not dict
        or set(marker) != expected_names
    ):
        _reject("runtime_pseudo_module_alias_invalid")
    rows: list[dict[str, Any]] = []
    for alias_name in sorted(expected_names, key=lambda value: value.encode("utf-8")):
        source = sealed[alias_name]
        observed = current[alias_name]
        owner_module = observed["owner_module"]
        owner_origin = owner_origins.get(owner_module)
        if (
            source.get("object") is not observed["object"]
            or marker[alias_name] is not observed["object"]
            or any(
                source.get(name) != observed[name]
                for name in (
                    "alias_name",
                    "owner_module",
                    "owner_attribute",
                    "object_kind",
                    "object_type",
                    "metadata_sha256",
                )
            )
            or not _is_sha256(owner_origin)
        ):
            _reject("runtime_pseudo_module_alias_invalid")
        body = {
            name: observed[name]
            for name in (
                "alias_name",
                "owner_module",
                "owner_attribute",
                "object_kind",
                "object_type",
                "metadata_sha256",
            )
        }
        body["owner_origin_sha256"] = owner_origin
        rows.append({**body, "row_sha256": contract.canonical_sha256(body)})
    return rows


def _runtime_live_critical_objects() -> dict[str, Any]:
    """Re-read critical transport identities from their live owners."""

    requests_module = sys.modules.get("requests")
    verifier_module = sys.modules.get(
        "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier"
    )
    sessions_module = sys.modules.get("requests.sessions")
    adapters_module = sys.modules.get("requests.adapters")
    if (
        type(requests_module) is not types.ModuleType
        or requests_module.__name__ != "requests"
        or type(verifier_module) is not types.ModuleType
        or verifier_module.__name__
        != "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier"
        or type(sessions_module) is not types.ModuleType
        or sessions_module.__name__ != "requests.sessions"
        or type(adapters_module) is not types.ModuleType
        or adapters_module.__name__ != "requests.adapters"
    ):
        _reject("runtime_critical_identity_invalid")
    session_class = getattr(requests_module, "Session", None)
    adapter_class = getattr(adapters_module, "HTTPAdapter", None)
    if (
        not inspect.isclass(session_class)
        or getattr(sessions_module, "Session", None) is not session_class
        or getattr(requests_module, "adapters", None) is not adapters_module
        or not inspect.isclass(adapter_class)
    ):
        _reject("runtime_critical_identity_invalid")
    for module_name in (
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
        "agent_benchmark.sec_filing_gemma_ollama",
    ):
        holder = sys.modules.get(module_name)
        if holder is not None and getattr(holder, "requests", None) is not requests_module:
            _reject("runtime_critical_identity_invalid")
    return {
        "requests_module": requests_module,
        "requests_session_class": session_class,
        "requests_adapter_class": adapter_class,
        "verifier_module": verifier_module,
    }


def _runtime_preload_repository_modules(root: Path) -> dict[str, Any]:
    """Load every allowed boundary and lazy numerical transport dependency."""

    if len(threading.enumerate()) != 1 or threading.current_thread() is not threading.main_thread():
        _reject("runtime_preload_thread_state_invalid")
    rows = [
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
        *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    ]
    names = [
        (
            path[: -len("/__init__.py")].replace("/", ".")
            if path.endswith("/__init__.py")
            else path[:-3].replace("/", ".")
        )
        for path in rows
    ]
    verifier_name = "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier"
    trap_hits: list[str] = []

    def network_trap(*_args: Any, **_kwargs: Any) -> Any:
        trap_hits.append("network")
        raise V315PreflightError("runtime_preload_network_attempted")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen
    original_builtin_open = builtins.open
    original_io_open = io.open

    private_tokens = (
        str((root / V38_PRIVATE_ROOT).resolve(strict=False)).casefold(),
        str((root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE)).resolve(strict=False)).casefold(),
    )

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = str(Path(file).resolve(strict=False)).casefold()
        except (TypeError, ValueError, OSError):
            candidate = ""
        if candidate and any(candidate.startswith(token) for token in private_tokens):
            trap_hits.append("private_body")
            raise V315PreflightError("runtime_preload_private_body_attempted")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = str(Path(file).resolve(strict=False)).casefold()
        except (TypeError, ValueError, OSError):
            candidate = ""
        if candidate and any(candidate.startswith(token) for token in private_tokens):
            trap_hits.append("private_body")
            raise V315PreflightError("runtime_preload_private_body_attempted")
        return original_io_open(file, *args, **kwargs)

    socket.socket.connect = network_trap
    socket.socket.connect_ex = network_trap
    socket.create_connection = network_trap
    urllib.request.urlopen = network_trap
    builtins.open = guarded_open
    io.open = guarded_io_open
    requests_send_original: Any | None = None
    try:
        verifier = importlib.import_module(
            "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier"
        )
        requests_module = verifier.load_allowed_requests()
        requests_send_original = requests_module.Session.send
        requests_module.Session.send = network_trap
        modules = {
            "agent_benchmark.sec_gemma_lean_science_v315_contract": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_contract"
            ),
            "agent_benchmark.sec_gemma_lean_science_v315_bridge": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_bridge"
            ),
            "agent_benchmark.sec_gemma_lean_science_v315_journal": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_journal"
            ),
            "agent_benchmark.sec_gemma_lean_science_v315_store": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_store"
            ),
            "agent_benchmark.sec_gemma_lean_science_v315_preflight": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_preflight"
            ),
            "agent_benchmark.sec_gemma_lean_science_v315_runner": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_science_v315_runner"
            ),
            "agent_benchmark.sec_audit_transport": importlib.import_module(
                "agent_benchmark.sec_audit_transport"
            ),
            "agent_benchmark.sec_filing_content": importlib.import_module(
                "agent_benchmark.sec_filing_content"
            ),
            "agent_benchmark.sec_filing_gemma_contract": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_contract"
            ),
            "agent_benchmark.sec_filing_gemma_corpus": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_corpus"
            ),
            "agent_benchmark.sec_filing_gemma_extractor_prompt": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_extractor_prompt"
            ),
            "agent_benchmark.sec_filing_gemma_extractor_schema": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_extractor_schema"
            ),
            "agent_benchmark.sec_filing_gemma_learner": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_learner"
            ),
            "agent_benchmark.sec_filing_gemma_market_acquirer": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_market_acquirer"
            ),
            "agent_benchmark.sec_filing_gemma_market_evidence": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_market_evidence"
            ),
            "agent_benchmark.sec_filing_gemma_market_source_bytes": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_market_source_bytes"
            ),
            "agent_benchmark.sec_filing_gemma_ollama": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_ollama"
            ),
            "agent_benchmark.sec_filing_gemma_preprocessor": importlib.import_module(
                "agent_benchmark.sec_filing_gemma_preprocessor"
            ),
            "agent_benchmark.sec_gemma_lean_v38_journal": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_v38_journal"
            ),
            "agent_benchmark.sec_gemma_lean_v38_source": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_v38_source"
            ),
            "agent_benchmark.sec_gemma_lean_v38_transport": importlib.import_module(
                "agent_benchmark.sec_gemma_lean_v38_transport"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_acquisition": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_acquisition"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_attempt": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_attempt"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_baseline": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_baseline"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_contract": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_contract"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_features": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_features"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_learner": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_learner"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_ledger": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_ledger"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_market_verifier": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_market_verifier"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_metrics": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_metrics"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_no_leverage": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_no_leverage"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_policy": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_policy"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_production": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_production"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_publisher": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_publisher"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_registry": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_registry"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_replay": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_replay"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_runner": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_runner"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_runtime": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_runtime"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier": verifier,
            "agent_benchmark.sec_gemma_online_risk_overlay_store": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_store"
            ),
            "agent_benchmark.sec_gemma_online_risk_overlay_vault": importlib.import_module(
                "agent_benchmark.sec_gemma_online_risk_overlay_vault"
            ),
            "agent_benchmark.sec_point_in_time": importlib.import_module(
                "agent_benchmark.sec_point_in_time"
            ),
            "agent_benchmark.sec_session_calendar": importlib.import_module(
                "agent_benchmark.sec_session_calendar"
            ),
            "agent_benchmark": importlib.import_module("agent_benchmark"),
        }
        if tuple(names) != tuple(modules):
            _reject("runtime_preload_identity_invalid")
        if modules[verifier_name] is not verifier:
            _reject("runtime_preload_identity_invalid")
        session = requests_module.Session()
        session.trust_env = False
        prepared = session.prepare_request(
            requests_module.Request(
                "GET", "https://example.invalid/v315-runtime-preload"
            )
        )
        adapter = session.get_adapter(prepared.url)
        if prepared.method != "GET" or prepared.url is None:
            _reject("runtime_preload_transport_invalid")
        session.close()
        https = http.client.HTTPSConnection(
            "example.invalid", context=ssl.create_default_context()
        )
        https.close()
        numpy = importlib.import_module("numpy")
        a = numpy.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=float)
        b = numpy.asarray([1.0, 0.0], dtype=float)
        # Exercise all lazy kernels used by the frozen learner on tiny fixed
        # arrays.  Results are deliberately discarded and never serialized.
        numpy.linalg.solve(a, b)
        numpy.linalg.lstsq(a, b, rcond=None)
        numpy.matmul(a, a)
        numpy.empty((2, 2), dtype=float)
        numpy.empty_like(a)
        numpy.ones((2, 2), dtype=float)
        numpy.full((2, 2), 1.0, dtype=float)
        numpy.zeros((2, 2), dtype=float)
        numpy.abs(a)
        numpy.max(a)
        numpy.maximum(a, 0.0)
        numpy.median(a, axis=0)
        numpy.mean(a, axis=0)
        numpy.logaddexp(a, 0.0)
        numpy.exp(a)
        numpy.clip(a, -1.0, 1.0)
        numpy.diag(b)
        numpy.column_stack((b, b))
        numpy.isfinite(a).all()
        numpy.isin(b, (0.0, 1.0)).all()
        numpy.any(a)
        timezone_rows = _runtime_timezone_material()
    except V315PreflightError:
        raise
    except Exception:
        _reject("runtime_preload_failed")
    finally:
        if requests_send_original is not None:
            requests_module.Session.send = requests_send_original
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection
        urllib.request.urlopen = original_urlopen
        builtins.open = original_builtin_open
        io.open = original_io_open
    if trap_hits:
        _reject("runtime_preload_effect_attempted")
    if len(threading.enumerate()) != 1:
        _reject("runtime_preload_thread_state_invalid")
    pseudo_alias_seal = _runtime_normalize_pseudo_module_aliases()
    production = modules.get(
        "agent_benchmark.sec_gemma_online_risk_overlay_production"
    )
    ollama = modules.get("agent_benchmark.sec_filing_gemma_ollama")
    for holder in (production, ollama):
        if holder is not None and getattr(holder, "requests", None) is not requests_module:
            _reject("runtime_requests_identity_invalid")
    live_critical = _runtime_live_critical_objects()
    if (
        live_critical["requests_module"] is not requests_module
        or live_critical["requests_session_class"] is not requests_module.Session
        or live_critical["requests_adapter_class"] is not type(adapter)
        or live_critical["verifier_module"] is not verifier
    ):
        _reject("runtime_critical_identity_invalid")
    return {
        "modules": modules,
        **live_critical,
        "timezone_rows": timezone_rows,
        "pseudo_alias_seal": pseudo_alias_seal,
    }


def _runtime_final_sys_path(root: Path) -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    parent = executable.parent
    initial = (
        "",
        str(parent / "python312.zip"),
        str(parent / "DLLs"),
        str(parent / "Lib"),
        str(parent),
    )
    derived = derive_final_sys_path(
        initial,
        python_executable=executable,
        repository_root=root,
        purelib_root=sysconfig.get_path("purelib"),
        platlib_root=sysconfig.get_path("platlib"),
    )
    try:
        observed = tuple(Path(item).resolve(strict=True) for item in sys.path)
    except (OSError, TypeError, ValueError):
        _reject("runtime_final_sys_path_invalid")
    if observed != derived["resolved_paths"] or any(
        name in sys.modules for name in ("site", "sitecustomize", "usercustomize")
    ):
        _reject("runtime_final_sys_path_invalid")
    return derived


def _runtime_current_environment(
    *, git_executable: Path, python_executable: Path
) -> dict[str, Any]:
    if set(os.environ) != set(RUNTIME_ENVIRONMENT_ALLOWLIST):
        _reject("runtime_environment_invalid")
    derived = build_reduced_runtime_environment(
        os.environ,
        git_executable=git_executable,
        python_executable=python_executable,
    )
    if derived["child_environment"] != dict(os.environ):
        _reject("runtime_environment_invalid")
    return derived


def _runtime_git_identity() -> tuple[Path, dict[str, Any]]:
    trusted = _runtime_active_preflight_context().get("trusted_git")
    if (
        isinstance(trusted, tuple)
        and len(trusted) == 3
        and isinstance(trusted[0], Path)
        and type(trusted[1]) is bytes
        and type(trusted[2]) is bytes
    ):
        executable, payload, version = trusted
    else:
        executable, payload, version = _resolve_authenticated_git_runtime()
    resolved_from_path = shutil.which("git", path=os.environ.get("PATH"))
    try:
        observed = Path(resolved_from_path).resolve(strict=True) if resolved_from_path else None
    except OSError:
        observed = None
    if observed != executable.resolve(strict=True):
        _reject("runtime_git_identity_invalid")
    return executable, {
        "basename": executable.name,
        "resolved_path_sha256": private_windows_path_sha256(executable),
        "byte_count": len(payload),
        "literal_sha256": hashlib.sha256(payload).hexdigest(),
        "version_output_sha256": hashlib.sha256(version).hexdigest(),
    }


def _build_execution_dependency_manifest(
    root: Path,
    *,
    repository_manifest: Mapping[str, Any],
    preload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the full current dependency closure after canonical preload."""

    before_modules = tuple(sorted(sys.modules, key=lambda value: value.encode("utf-8")))
    before_images = _runtime_loaded_image_paths()
    flags, flags_sha256 = _runtime_python_flags()
    del flags
    git_executable, git_row = _runtime_git_identity()
    executable = Path(sys.executable).resolve(strict=True)
    executable_payload = _read_regular_file(
        executable, maximum=None, code="runtime_python_identity_invalid"
    )
    final_path = _runtime_final_sys_path(root)
    environment = _runtime_current_environment(
        git_executable=git_executable,
        python_executable=executable,
    )
    launcher = build_launcher_profile(
        python_executable_sha256=hashlib.sha256(executable_payload).hexdigest(),
        bootstrap_bytes=SCIENTIFIC_BOOTSTRAP_BYTES,
    )
    install_parent = executable.parent.resolve(strict=True)
    sealed_roots: list[Path] = []
    for candidate in (
        install_parent,
        install_parent / "Lib",
        install_parent / "DLLs",
        Path(sysconfig.get_path("purelib")),
        Path(sysconfig.get_path("platlib")),
    ):
        resolved = candidate.resolve(strict=True)
        if resolved not in sealed_roots:
            sealed_roots.append(resolved)
    owners, distribution_inventory = _runtime_distribution_inventory(sealed_roots)
    builtins_rows, module_rows, distribution_rows = _runtime_module_inventory(
        repository_root=root,
        repository_manifest=repository_manifest,
        distribution_owners=owners,
        distribution_rows=distribution_inventory,
        sealed_install_roots=sealed_roots,
    )
    owner_origins = {
        row["module_name"]: contract.canonical_sha256(row)
        for row in (*builtins_rows, *module_rows)
    }
    raw_pseudo_seal = preload.get("pseudo_alias_seal")
    pseudo_alias_seal = (
        raw_pseudo_seal
        if isinstance(raw_pseudo_seal, Mapping)
        else _runtime_pseudo_module_alias_seal(require_present=False)
    )
    pseudo_alias_rows = _runtime_pseudo_module_alias_rows(
        pseudo_alias_seal,
        owner_origins=owner_origins,
    )
    binary_rows = _runtime_binary_rows(before_images)
    python_dlls = [
        (path, row)
        for path, row in zip(before_images, binary_rows, strict=True)
        if path.name.casefold() == "python312.dll"
    ]
    if sys.platform == "win32" and len(python_dlls) != 1:
        _reject("runtime_python_dll_invalid")
    if python_dlls:
        python_dll_path, python_dll_row = python_dlls[0]
    else:
        # Non-Windows unit environments bind the executable as the runtime
        # image; production is Windows and always takes the strict branch.
        python_dll_path = executable
        python_dll_row = next(
            row
            for row in binary_rows
            if row["resolved_path_sha256"] == private_windows_path_sha256(executable)
        )
    timezone_rows = copy.deepcopy(preload["timezone_rows"])
    python_runtime = {
        "executable_basename": executable.name,
        "executable_path_sha256": private_windows_path_sha256(executable),
        "executable_bytes": len(executable_payload),
        "executable_sha256": hashlib.sha256(executable_payload).hexdigest(),
        "python_dll_basename": python_dll_path.name,
        "python_dll_path_sha256": python_dll_row["resolved_path_sha256"],
        "python_dll_bytes": python_dll_row["byte_count"],
        "python_dll_sha256": python_dll_row["literal_sha256"],
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "os_name": os.name,
        "sys_platform": sys.platform,
        "launcher_profile_sha256": launcher["launcher_profile_sha256"],
        "bootstrap_sha256": SCIENTIFIC_BOOTSTRAP_SHA256,
        "flags_sha256": flags_sha256,
        "process_environment_sha256": environment["process_environment_sha256"],
    }
    counts = {
        "builtin_or_frozen_modules": len(builtins_rows),
        "module_files": len(module_rows),
        "distributions": len(distribution_rows),
        "loaded_binaries": len(binary_rows),
        "timezone_files": len(timezone_rows),
    }
    body = {
        "schema_version": EXECUTION_DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        "python_runtime": python_runtime,
        "git_runtime": git_row,
        "builtin_or_frozen_modules": builtins_rows,
        "module_files": module_rows,
        "distributions": distribution_rows,
        "loaded_binaries": binary_rows,
        "timezone_files": timezone_rows,
        "normalized_pseudo_module_aliases": pseudo_alias_rows,
        "sys_path_sha256": final_path["final_sys_path_sha256"],
        "counts": counts,
    }
    after_modules = tuple(sorted(sys.modules, key=lambda value: value.encode("utf-8")))
    after_images = _runtime_loaded_image_paths()
    if before_modules != after_modules or tuple(
        private_windows_path_sha256(path) for path in before_images
    ) != tuple(private_windows_path_sha256(path) for path in after_images):
        _reject("runtime_dependency_snapshot_changed")
    return {
        **body,
        "execution_dependency_manifest_sha256": contract.canonical_sha256(body),
    }


def validate_execution_dependency_manifest(value: Any) -> dict[str, Any]:
    code = "execution_dependency_manifest_invalid"
    item = _strict_mapping(value, _EXECUTION_DEPENDENCY_FIELDS, code=code)
    _validate_self_hash(item, "execution_dependency_manifest_sha256", code=code)
    python_runtime = _strict_mapping(item["python_runtime"], _PYTHON_RUNTIME_FIELDS, code=code)
    git_runtime = _strict_mapping(item["git_runtime"], _GIT_RUNTIME_FIELDS, code=code)
    row_specs = (
        ("builtin_or_frozen_modules", _BUILTIN_RUNTIME_ROW_FIELDS),
        ("module_files", _MODULE_RUNTIME_ROW_FIELDS),
        ("distributions", _DISTRIBUTION_RUNTIME_ROW_FIELDS),
        ("loaded_binaries", _BINARY_RUNTIME_ROW_FIELDS),
        ("timezone_files", _TIMEZONE_RUNTIME_ROW_FIELDS),
    )
    parsed: dict[str, list[dict[str, Any]]] = {}
    if item["schema_version"] != EXECUTION_DEPENDENCY_MANIFEST_SCHEMA_VERSION:
        _reject(code)
    for name, fields in row_specs:
        raw = item[name]
        if type(raw) is not list:
            _reject(code)
        parsed[name] = [_strict_mapping(row, fields, code=code) for row in raw]
    raw_pseudo_aliases = item["normalized_pseudo_module_aliases"]
    if type(raw_pseudo_aliases) is not list:
        _reject(code)
    pseudo_aliases = [
        _strict_mapping(row, _PSEUDO_MODULE_ALIAS_ROW_FIELDS, code=code)
        for row in raw_pseudo_aliases
    ]
    counts = _strict_mapping(
        item["counts"],
        frozenset(name for name, _fields in row_specs),
        code=code,
    )
    flag_names = (
        "debug", "inspect", "interactive", "optimize", "dont_write_bytecode",
        "no_user_site", "no_site", "ignore_environment", "verbose",
        "bytes_warning", "quiet", "hash_randomization", "isolated",
        "dev_mode", "utf8_mode", "warn_default_encoding", "safe_path",
        "int_max_str_digits",
    )
    flag_values = (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, False, 1, 0, False, 4300)
    expected_flags_sha256 = contract.canonical_sha256(
        dict(zip(flag_names, flag_values, strict=True))
    )
    expected_launcher = build_launcher_profile(
        python_executable_sha256=python_runtime["executable_sha256"],
        bootstrap_bytes=SCIENTIFIC_BOOTSTRAP_BYTES,
    )
    if (
        any(
            type(python_runtime[name]) is not str or not python_runtime[name]
            for name in (
                "executable_basename", "python_dll_basename", "python_version",
                "cache_tag", "os_name", "sys_platform",
            )
        )
        or python_runtime["executable_basename"]
        != contract.QUALIFICATION_EXECUTABLE_BASENAME
        or python_runtime["python_dll_basename"].casefold() != "python312.dll"
        or python_runtime["python_version"] != contract.QUALIFICATION_PYTHON_VERSION
        or python_runtime["cache_tag"] != contract.QUALIFICATION_PYTHON_CACHE_TAG
        or python_runtime["os_name"] != contract.QUALIFICATION_OS_NAME
        or python_runtime["sys_platform"] != contract.QUALIFICATION_SYS_PLATFORM
        or type(python_runtime["executable_bytes"]) is not int
        or python_runtime["executable_bytes"] <= 0
        or type(python_runtime["python_dll_bytes"]) is not int
        or python_runtime["python_dll_bytes"] <= 0
        or python_runtime["bootstrap_sha256"] != SCIENTIFIC_BOOTSTRAP_SHA256
        or python_runtime["flags_sha256"] != expected_flags_sha256
        or python_runtime["launcher_profile_sha256"]
        != expected_launcher["launcher_profile_sha256"]
        or type(git_runtime["basename"]) is not str
        or git_runtime["basename"].casefold() != "git.exe"
        or type(git_runtime["byte_count"]) is not int
        or git_runtime["byte_count"] <= 0
        or any(
            type(counts[name]) is not int
            or counts[name] != len(parsed[name])
            for name, _fields in row_specs
        )
        or not _is_sha256(item["sys_path_sha256"])
        or not all(
            _is_sha256(python_runtime[name])
            for name in (
                "executable_path_sha256",
                "executable_sha256",
                "python_dll_path_sha256",
                "python_dll_sha256",
                "launcher_profile_sha256",
                "bootstrap_sha256",
                "flags_sha256",
                "process_environment_sha256",
            )
        )
        or not all(_is_sha256(git_runtime[name]) for name in ("resolved_path_sha256", "literal_sha256", "version_output_sha256"))
    ):
        _reject(code)
    builtin_keys: list[tuple[str, str]] = []
    for row in parsed["builtin_or_frozen_modules"]:
        if (
            type(row["module_name"]) is not str
            or not row["module_name"]
            or row["origin_kind"] not in {"builtin", "frozen"}
        ):
            _reject(code)
        builtin_keys.append((row["module_name"], row["origin_kind"]))
    if builtin_keys != sorted(
        builtin_keys, key=lambda pair: (pair[0].encode("utf-8"), pair[1])
    ) or len(builtin_keys) != len(set(builtin_keys)):
        _reject(code)
    module_keys: list[tuple[str, str, str, str]] = []
    for row in parsed["module_files"]:
        if (
            type(row["module_name"]) is not str
            or not row["module_name"]
            or
            row["origin_kind"] not in {
                "stdlib_source",
                "stdlib_bytecode",
                "distribution_source",
                "distribution_bytecode",
                "zip_member",
                "extension_module",
            }
            or not _is_sha256(row["resolved_path_sha256"])
            or not _is_sha256(row["literal_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 0
            or (
                row["origin_kind"] == "zip_member"
                and (
                    type(row["member_name"]) is not str
                    or not row["member_name"]
                    or "\\" in row["member_name"]
                    or any(part in {"", ".", ".."} for part in row["member_name"].split("/"))
                )
            )
            or (
                row["origin_kind"] != "zip_member"
                and row["member_name"] is not None
            )
            or (
                row["origin_kind"].startswith("stdlib_")
                and row["distribution_name"] is not None
            )
            or (
                row["origin_kind"].startswith("distribution_")
                and (
                    type(row["distribution_name"]) is not str
                    or _normalized_distribution_name(row["distribution_name"])
                    != row["distribution_name"]
                )
            )
            or (
                row["distribution_name"] is not None
                and type(row["distribution_name"]) is not str
            )
        ):
            _reject(code)
        module_keys.append(
            (
                row["module_name"], row["origin_kind"],
                row["resolved_path_sha256"], row["member_name"] or "",
            )
        )
    if module_keys != sorted(
        module_keys,
        key=lambda pair: (pair[0].encode("utf-8"), pair[1], pair[2], pair[3]),
    ) or len(module_keys) != len(set(module_keys)):
        _reject(code)
    if set(name for name, _kind in builtin_keys) & set(
        key[0] for key in module_keys
    ):
        _reject(code)
    origin_hashes = {
        row["module_name"]: contract.canonical_sha256(row)
        for row in (
            *parsed["builtin_or_frozen_modules"],
            *parsed["module_files"],
        )
    }
    pseudo_specs = {
        alias_name: (owner_module, owner_attribute, object_kind, object_type)
        for (
            alias_name,
            owner_module,
            owner_attribute,
            object_kind,
            object_type,
        ) in _RUNTIME_PSEUDO_MODULE_ALIAS_SPECS
    }
    if [row["alias_name"] for row in pseudo_aliases] != sorted(
        pseudo_specs, key=lambda value: value.encode("utf-8")
    ):
        _reject(code)
    for row in pseudo_aliases:
        expected = pseudo_specs.get(row["alias_name"])
        if expected is None:
            _reject(code)
        owner_module, owner_attribute, object_kind, object_type = expected
        metadata = {
            "alias_name": row["alias_name"],
            "object_name": row["alias_name"],
            "spec_is_null": True,
            "package_is_null": True,
            "loader_is_null": True,
            "file_attribute_present": False,
        }
        body = {name: row[name] for name in _PSEUDO_MODULE_ALIAS_ROW_FIELDS if name != "row_sha256"}
        if (
            row["owner_module"] != owner_module
            or row["owner_attribute"] != owner_attribute
            or row["object_kind"] != object_kind
            or row["object_type"] != object_type
            or row["metadata_sha256"] != contract.canonical_sha256(metadata)
            or row["owner_origin_sha256"] != origin_hashes.get(owner_module)
            or row["row_sha256"] != contract.canonical_sha256(body)
        ):
            _reject(code)
    distribution_keys: list[tuple[str, str]] = []
    for row in parsed["distributions"]:
        if (
            type(row["normalized_name"]) is not str
            or _normalized_distribution_name(row["normalized_name"])
            != row["normalized_name"]
            or type(row["version"]) is not str
            or not row["version"]
            or not _is_sha256(row["record_sha256"])
            or type(row["file_count"]) is not int
            or row["file_count"] <= 0
            or not _is_sha256(row["ordered_files_sha256"])
        ):
            _reject(code)
        distribution_keys.append((row["normalized_name"], row["version"]))
    if distribution_keys != sorted(
        distribution_keys,
        key=lambda pair: (pair[0].encode("utf-8"), pair[1].encode("utf-8")),
    ) or len(distribution_keys) != len(set(distribution_keys)):
        _reject(code)
    distribution_names = {name for name, _version in distribution_keys}
    if any(
        row["distribution_name"] is not None
        and row["distribution_name"] not in distribution_names
        for row in parsed["module_files"]
    ):
        _reject(code)
    binary_keys: list[tuple[str, str]] = []
    for row in parsed["loaded_binaries"]:
        if (
            type(row["basename"]) is not str
            or not row["basename"]
            or any(separator in row["basename"] for separator in ("/", "\\"))
            or not _is_sha256(row["resolved_path_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
            or not _is_sha256(row["literal_sha256"])
        ):
            _reject(code)
        binary_keys.append((row["basename"], row["resolved_path_sha256"]))
    if binary_keys != sorted(
        binary_keys,
        key=lambda pair: (pair[0].casefold().encode("utf-8"), pair[1]),
    ) or len({token for _name, token in binary_keys}) != len(binary_keys):
        _reject(code)
    timezone_keys: list[tuple[str, str, str]] = []
    for row in parsed["timezone_files"]:
        if (
            row["zone_key"] not in _RUNTIME_TIMEZONE_KEYS
            or type(row["provider"]) is not str
            or not row["provider"]
            or (
                row["resolved_path_sha256"] is None
                and not row["provider"].startswith("tzdata:")
            )
            or (
                row["resolved_path_sha256"] is not None
                and not _is_sha256(row["resolved_path_sha256"])
            )
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
            or not _is_sha256(row["literal_sha256"])
        ):
            _reject(code)
        timezone_keys.append(
            (row["zone_key"], row["provider"], row["resolved_path_sha256"] or "")
        )
    if (
        {row["zone_key"] for row in parsed["timezone_files"]}
        != set(_RUNTIME_TIMEZONE_KEYS)
        or timezone_keys
        != sorted(
            timezone_keys,
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8"), item[2]),
        )
        or len(timezone_keys) != len(set(timezone_keys))
    ):
        _reject(code)
    dll_matches = [
        row
        for row in parsed["loaded_binaries"]
        if row["resolved_path_sha256"] == python_runtime["python_dll_path_sha256"]
    ]
    if (
        len(dll_matches) != 1
        or dll_matches[0]["basename"] != python_runtime["python_dll_basename"]
        or dll_matches[0]["byte_count"] != python_runtime["python_dll_bytes"]
        or dll_matches[0]["literal_sha256"] != python_runtime["python_dll_sha256"]
    ):
        _reject(code)
    return {
        **item,
        "python_runtime": python_runtime,
        "git_runtime": git_runtime,
        **parsed,
        "normalized_pseudo_module_aliases": pseudo_aliases,
        "counts": counts,
    }


class _RuntimeFingerprintContext:
    """Deterministic tagged encoder with process-local caches and cycle checks."""

    def __init__(
        self,
        *,
        repository: Mapping[str, Any],
        dependency: Mapping[str, Any],
    ) -> None:
        repo = validate_repository_runtime_manifest(repository)
        dep = validate_execution_dependency_manifest(dependency)
        self.repo_rows = {row["module_name"]: row for row in repo["modules"]}
        self.repo_origins = {
            name: contract.canonical_sha256(row)
            for name, row in self.repo_rows.items()
        }
        self.dependency_origins: dict[str, str] = {}
        self.dependency_files: dict[str, dict[str, Any]] = {}
        for row in dep["builtin_or_frozen_modules"]:
            self.dependency_origins[row["module_name"]] = contract.canonical_sha256(row)
        for row in dep["module_files"]:
            name = row["module_name"]
            if name in self.dependency_origins:
                _reject("runtime_fingerprint_origin_duplicate")
            self.dependency_origins[name] = contract.canonical_sha256(row)
            self.dependency_files[name] = row
        self.binary_by_path = {
            row["resolved_path_sha256"]: row
            for row in dep["loaded_binaries"]
        }
        self.python_dll_sha256 = dep["python_runtime"]["python_dll_sha256"]
        self.timezones = {
            row["zone_key"]: row["literal_sha256"]
            for row in dep["timezone_files"]
        }
        self.code_cache: dict[tuple[int, str], bytes] = {}
        self.value_cache: dict[int, tuple[Any, bytes]] = {}
        self.active: set[int] = set()
        self.identity_sentinels: dict[int, tuple[Any, str]] = {}
        self.lock_owners: dict[int, tuple[Any, str]] = {}
        self.weak_registry_owners: dict[int, tuple[Any, str]] = {}
        self.weak_self_refs: dict[int, tuple[Any, Any]] = {}
        self.relative_path_bindings: dict[
            int, tuple[Any, tuple[tuple[str, str], ...], str]
        ] = {}
        self.typing_alias_bindings: dict[
            int, tuple[Any, tuple[tuple[str, str], ...], str]
        ] = {}
        self.owner_slots: dict[int, tuple[Any, tuple[tuple[str, ...], ...]]] = {}
        self.owner_slot_hashes: dict[int, tuple[Any, str]] = {}
        self.referenced_callables: dict[int, Any] = {}

    @staticmethod
    def _pack(tag: str, fields: Sequence[bytes] = ()) -> bytes:
        output = bytearray(tag.encode("ascii"))
        output.extend(b"[")
        for field in fields:
            output.extend(str(len(field)).encode("ascii"))
            output.extend(b":")
            output.extend(field)
        output.extend(b"]")
        return bytes(output)

    @staticmethod
    def _text(value: str) -> bytes:
        return value.encode("utf-8")

    def module_reference(self, module: types.ModuleType) -> bytes:
        name = getattr(module, "__name__", None)
        if type(name) is not str:
            _reject("runtime_fingerprint_module_invalid")
        if name in self.repo_origins:
            return self._pack(
                "repository_module",
                (self._text(name), self._text(self.repo_origins[name])),
            )
        origin = self.dependency_origins.get(name)
        if origin is None:
            _reject("runtime_fingerprint_module_origin_missing")
        return self._pack("external_module", (self._text(name), self._text(origin)))

    def callable_reference(self, value: Any) -> bytes:
        if not _runtime_is_supported_callable(value):
            _reject("runtime_fingerprint_callable_type_unsupported")
        identifier = id(value)
        prior = self.referenced_callables.setdefault(identifier, value)
        if prior is not value:
            _reject("runtime_fingerprint_callable_identity_invalid")
        module = getattr(value, "__module__", None)
        qualname = self.callable_qualified_name(value)
        if type(module) is not str:
            module = type(value).__module__
        owner_slots = self.owner_slot_hashes.get(identifier)
        if owner_slots is None or owner_slots[0] is not value:
            _reject("runtime_fingerprint_callable_owner_slots_missing")
        return self._pack(
            "callable_reference",
            (
                self._text(module),
                self._text(qualname),
                self._text(_runtime_callable_kind(value)),
                self._text(owner_slots[1]),
            ),
        )

    def register_callable_qualification(self, value: Any, qualname: str) -> None:
        if (
            not _runtime_is_supported_callable(value)
            or type(qualname) is not str
            or qualname != self.callable_qualified_name(value)
        ):
            _reject("runtime_fingerprint_callable_qualification_invalid")

    def callable_qualified_name(self, value: Any) -> str:
        qualname = getattr(value, "__qualname__", None)
        if type(qualname) is not str:
            qualname = type(value).__qualname__
        if not qualname:
            _reject("runtime_fingerprint_callable_qualification_invalid")
        return qualname

    def owner_token(self, owner_module: str) -> str:
        row = self.repo_rows.get(owner_module)
        if row is not None:
            return f"repo:{row['relative_path']}"
        file_row = self.dependency_files.get(owner_module)
        if file_row is None:
            return f"builtin:{owner_module}"
        token = f"path-sha256:{file_row['resolved_path_sha256']}"
        if file_row["member_name"] is not None:
            token += f"!{file_row['member_name']}"
        return token

    def code(self, value: types.CodeType, *, owner_module: str) -> bytes:
        key = (id(value), owner_module)
        cached = self.code_cache.get(key)
        if cached is not None:
            return cached
        identifier = id(value)
        if identifier in self.active:
            _reject("runtime_fingerprint_cycle")
        self.active.add(identifier)
        try:
            constants = self._pack(
                "code_constants",
                tuple(
                    self.code(item, owner_module=owner_module)
                    if isinstance(item, types.CodeType)
                    else self.value(item)
                    for item in value.co_consts
                ),
            )
            integer_fields = (
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_nlocals,
                value.co_stacksize,
                value.co_flags,
                value.co_firstlineno,
            )
            encoded = self._pack(
                "code",
                (
                    self._text(self.owner_token(owner_module)),
                    self._text(value.co_qualname),
                    *(self._text(str(item)) for item in integer_fields),
                    value.co_code,
                    constants,
                    self.value(tuple(value.co_names)),
                    self.value(tuple(value.co_varnames)),
                    self.value(tuple(value.co_freevars)),
                    self.value(tuple(value.co_cellvars)),
                    value.co_linetable,
                    value.co_exceptiontable,
                ),
            )
        finally:
            self.active.remove(identifier)
        self.code_cache[key] = encoded
        return encoded

    def _lock(self, value: Any) -> bytes:
        owner = self.lock_owners.get(id(value))
        if owner is None or owner[0] is not value:
            _reject("runtime_fingerprint_lock_owner_missing")
        if type(value) is _RUNTIME_LOCK_TYPE:
            if value.locked() is not False:
                _reject("runtime_fingerprint_lock_state_invalid")
            kind = "_thread.lock"
        elif type(value) is _RUNTIME_RLOCK_TYPE:
            if value._is_owned() is not False or not value.acquire(False):
                _reject("runtime_fingerprint_lock_state_invalid")
            value.release()
            kind = "_thread.RLock"
        else:
            _reject("runtime_fingerprint_lock_type_invalid")
        return self._pack(
            "synchronization",
            (
                self._text(owner[1]),
                self._text(kind),
                self._text(sys.version),
                self._text("builtin:_thread"),
                self._text("initial_unlocked_or_unowned"),
            ),
        )

    def _weak_registry(self, value: Any) -> bytes:
        owner = self.weak_registry_owners.get(id(value))
        if (
            owner is None
            or owner[0] is not value
            or type(value) is not weakref.WeakKeyDictionary
        ):
            _reject("runtime_fingerprint_weak_registry_owner_missing")
        details = _runtime_validate_weak_registry(value)
        callback = details["callback"]
        self_ref = details["self_ref"]
        registered_ref = self.weak_self_refs.get(id(self_ref))
        if (
            registered_ref is None
            or registered_ref[0] is not self_ref
            or registered_ref[1] is not value
        ):
            _reject("runtime_fingerprint_weak_registry_self_ref_missing")
        return self._pack(
            "weak_key_registry",
            (
                self._text(owner[1]),
                self._text("weakref.WeakKeyDictionary"),
                self._text(self.dependency_origins["weakref"]),
                self.callable_reference(callback),
                self._text(
                    _runtime_component_sha256(
                        self.code(callback.__code__, owner_module="weakref")
                    )
                ),
                self.value(callback.__defaults__),
            ),
        )

    def _weak_self_reference(self, value: weakref.ReferenceType[Any]) -> bytes:
        registered = self.weak_self_refs.get(id(value))
        if registered is None or registered[0] is not value:
            _reject("runtime_fingerprint_weak_reference_unsupported")
        registry = registered[1]
        owner = self.weak_registry_owners.get(id(registry))
        if (
            owner is None
            or owner[0] is not registry
            or value() is not registry
            or value.__callback__ is not None
        ):
            _reject("runtime_fingerprint_weak_reference_invalid")
        return self._pack(
            "weak_registry_self_ref",
            (self._text(owner[1]), self._text("referent")),
        )

    def _relative_path(self, value: Path) -> bytes:
        registered = self.relative_path_bindings.get(id(value))
        if registered is None or registered[0] is not value:
            _reject("runtime_fingerprint_relative_path_invalid")
        _item, bindings, posix_value = registered
        return self._pack(
            "relative_path",
            (
                self._text("pathlib.WindowsPath"),
                contract.canonical_json_bytes([list(binding) for binding in bindings]),
                contract.canonical_json_bytes(list(value.parts)),
                self._text(posix_value),
            ),
        )

    def _typing_alias(self, value: Any) -> bytes:
        registered = self.typing_alias_bindings.get(id(value))
        if registered is None or registered[0] is not value:
            _reject("runtime_fingerprint_typing_alias_invalid")
        owner_slots = self.owner_slot_hashes.get(id(value))
        if (
            owner_slots is None
            or owner_slots[0] is not value
            or not _is_sha256(owner_slots[1])
        ):
            _reject("runtime_fingerprint_typing_alias_owner_slots_invalid")
        _item, bindings, alias_name = registered
        state = _runtime_validate_typing_alias(value, alias_name=alias_name)
        return self._pack(
            "typing_alias",
            (
                self._text(alias_name),
                self._text(owner_slots[1]),
                contract.canonical_json_bytes([list(binding) for binding in bindings]),
                self._text(str(state["_nparams"])),
                self._text(hashlib.sha256(state["__doc__"].encode("utf-8")).hexdigest()),
                self.callable_reference(type(value)),
                self.callable_reference(state["__origin__"]),
            ),
        )

    def value(self, value: Any) -> bytes:
        if value is None:
            return self._pack("null")
        if value is Ellipsis:
            return self._pack("ellipsis")
        if value is NotImplemented:
            return self._pack("not_implemented")
        if type(value) is bool:
            return self._pack("bool", (b"1" if value else b"0",))
        if type(value) is int:
            return self._pack("integer", (str(value).encode("ascii"),))
        if type(value) is float:
            return self._pack("float", (value.hex().encode("ascii"),))
        if type(value) is complex:
            return self._pack(
                "complex",
                (value.real.hex().encode("ascii"), value.imag.hex().encode("ascii")),
            )
        if type(value) is str:
            return self._pack("string", (value.encode("utf-8"),))
        if type(value) is bytes:
            return self._pack("bytes", (value,))
        if isinstance(value, types.CodeType):
            _reject("runtime_fingerprint_unowned_code")
        if isinstance(value, types.ModuleType):
            return self.module_reference(value)
        if _runtime_is_supported_callable(value):
            return self.callable_reference(value)
        if _runtime_is_typing_semantic(value):
            return self._typing_alias(value)
        if type(value) is weakref.ReferenceType:
            return self._weak_self_reference(value)
        if callable(value):
            _reject("runtime_fingerprint_stateful_callable_unsupported")
        if type(value) in {_RUNTIME_LOCK_TYPE, _RUNTIME_RLOCK_TYPE}:
            return self._lock(value)
        if type(value) is weakref.WeakKeyDictionary:
            return self._weak_registry(value)
        sentinel = self.identity_sentinels.get(id(value))
        if sentinel is not None:
            if sentinel[0] is not value or type(value) is not object:
                _reject("runtime_fingerprint_identity_sentinel_invalid")
            return self._pack(
                "identity_sentinel",
                (self._text("builtins.object"), self._text(sentinel[1])),
            )
        identifier = id(value)
        cached = self.value_cache.get(identifier)
        if cached is not None:
            cached_object, cached_bytes = cached
            if cached_object is not value:
                _reject("runtime_fingerprint_cache_identity_invalid")
            return cached_bytes
        if identifier in self.active:
            _reject("runtime_fingerprint_cycle")
        self.active.add(identifier)
        try:
            if type(value) is list:
                encoded = self._pack("list", tuple(self.value(item) for item in value))
            elif type(value) is tuple:
                encoded = self._pack("tuple", tuple(self.value(item) for item in value))
            elif isinstance(value, types.MappingProxyType):
                encoded = self._mapping(value, "mapping_proxy")
            elif isinstance(value, Mapping):
                encoded = self._mapping(value, "mapping")
            elif type(value) is set:
                if any(_runtime_value_has_graph_identity(item) for item in value):
                    _reject("runtime_fingerprint_identity_set_unsupported")
                encoded = self._pack(
                    "set", tuple(sorted(self.value(item) for item in value))
                )
            elif type(value) is frozenset:
                if any(_runtime_value_has_graph_identity(item) for item in value):
                    _reject("runtime_fingerprint_identity_set_unsupported")
                encoded = self._pack(
                    "frozenset", tuple(sorted(self.value(item) for item in value))
                )
            elif type(value) is range:
                encoded = self._pack(
                    "range",
                    tuple(str(item).encode("ascii") for item in (value.start, value.stop, value.step)),
                )
            elif type(value) is slice:
                encoded = self._pack(
                    "slice", tuple(self.value(item) for item in (value.start, value.stop, value.step))
                )
            elif isinstance(value, re.Pattern):
                encoded = self._pack(
                    "regex", (self.value(value.pattern), self.value(value.flags))
                )
            elif isinstance(value, enum.Enum):
                encoded = self._pack(
                    "enum",
                    (
                        self.callable_reference(type(value)),
                        self._text(value.name),
                        self.value(value.value),
                    ),
                )
            elif isinstance(value, zoneinfo.ZoneInfo):
                digest = self.timezones.get(value.key)
                if digest is None:
                    _reject("runtime_fingerprint_timezone_missing")
                encoded = self._pack(
                    "zoneinfo", (self._text(value.key), self._text(digest))
                )
            elif type(value) in {
                _datetime.date,
                _datetime.datetime,
                _datetime.time,
            }:
                encoded = self._pack(
                    f"datetime_{type(value).__name__}",
                    (self._text(value.isoformat()),),
                )
            elif type(value) is _datetime.timedelta:
                encoded = self._pack(
                    "datetime_timedelta",
                    tuple(
                        str(item).encode("ascii")
                        for item in (value.days, value.seconds, value.microseconds)
                    ),
                )
            elif isinstance(value, _datetime.timezone):
                offset = value.utcoffset(None)
                encoded = self._pack(
                    "fixed_timezone",
                    (self.value(offset), self.value(value.tzname(None))),
                )
            elif isinstance(value, Path):
                if value.is_absolute():
                    encoded = self._pack(
                        "path",
                        (self._text(private_windows_path_sha256(value)),),
                    )
                else:
                    encoded = self._relative_path(value)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                fields = tuple(
                    self._pack(
                        "dataclass_field",
                        (self._text(field.name), self.value(getattr(value, field.name))),
                    )
                    for field in dataclasses.fields(value)
                )
                encoded = self._pack(
                    "dataclass", (self.callable_reference(type(value)), *fields)
                )
            elif isinstance(value, types.SimpleNamespace):
                encoded = self._mapping(vars(value), "namespace")
            else:
                _reject("runtime_fingerprint_semantic_value_unsupported")
        finally:
            self.active.remove(identifier)
        self.value_cache[identifier] = (value, encoded)
        return encoded

    def _mapping(self, value: Mapping[Any, Any], tag: str) -> bytes:
        for key in value:
            try:
                _runtime_owner_free_value(key)
            except V315PreflightError:
                _reject("runtime_fingerprint_identity_mapping_key_unsupported")
        pairs = sorted(
            (self.value(key), self.value(child)) for key, child in value.items()
        )
        return self._pack(
            tag,
            tuple(self._pack("pair", pair) for pair in pairs),
        )


def _runtime_callable_kind(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value):
        return "function"
    return "native_callable"


def _runtime_is_supported_callable(value: Any) -> bool:
    return (
        inspect.isfunction(value)
        or inspect.isclass(value)
        or inspect.isbuiltin(value)
        or inspect.ismethoddescriptor(value)
        or isinstance(
            value,
            (
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.WrapperDescriptorType,
                types.MethodDescriptorType,
                types.ClassMethodDescriptorType,
                types.MethodWrapperType,
            ),
        )
    )


def _runtime_is_typing_semantic(value: Any) -> bool:
    """Recognize exact immutable Python 3.12 typing aliases, not callables."""

    return (
        type(value).__module__ == "typing"
        and type(value).__qualname__ == "_SpecialGenericAlias"
        and type(getattr(value, "_name", None)) is str
        and isinstance(getattr(value, "__origin__", None), type)
    )


def _runtime_owner_free_value(value: Any) -> bytes:
    """Encode only values that can safely name an identity-bearing map value."""

    pack = _RuntimeFingerprintContext._pack
    if value is None:
        return pack("null")
    if type(value) is bool:
        return pack("bool", (b"1" if value else b"0",))
    if type(value) is int:
        return pack("integer", (str(value).encode("ascii"),))
    if type(value) is float:
        return pack("float", (value.hex().encode("ascii"),))
    if type(value) is complex:
        return pack(
            "complex",
            (value.real.hex().encode("ascii"), value.imag.hex().encode("ascii")),
        )
    if type(value) is str:
        return pack("string", (value.encode("utf-8"),))
    if type(value) is bytes:
        return pack("bytes", (value,))
    if type(value) is range:
        return pack(
            "range",
            tuple(
                str(item).encode("ascii")
                for item in (value.start, value.stop, value.step)
            ),
        )
    if type(value) is slice:
        return pack(
            "slice",
            tuple(
                _runtime_owner_free_value(item)
                for item in (value.start, value.stop, value.step)
            ),
        )
    if type(value) in {_datetime.date, _datetime.datetime, _datetime.time}:
        return pack(
            f"datetime_{type(value).__name__}",
            (value.isoformat().encode("utf-8"),),
        )
    if type(value) is _datetime.timedelta:
        return pack(
            "datetime_timedelta",
            tuple(
                str(item).encode("ascii")
                for item in (value.days, value.seconds, value.microseconds)
            ),
        )
    if type(value) is _datetime.timezone:
        return pack(
            "fixed_timezone",
            (
                _runtime_owner_free_value(value.utcoffset(None)),
                _runtime_owner_free_value(value.tzname(None)),
            ),
        )
    _reject("runtime_owner_free_value_unsupported")


def _runtime_value_has_graph_identity(value: Any) -> bool:
    try:
        _runtime_owner_free_value(value)
    except V315PreflightError:
        if type(value) in {list, tuple, set, frozenset}:
            return any(_runtime_value_has_graph_identity(item) for item in value)
        if isinstance(value, (types.MappingProxyType, Mapping)):
            for key, child in value.items():
                try:
                    _runtime_owner_free_value(key)
                except V315PreflightError:
                    return True
                if _runtime_value_has_graph_identity(child):
                    return True
            return False
        if isinstance(value, types.SimpleNamespace):
            return _runtime_value_has_graph_identity(vars(value))
        if isinstance(value, re.Pattern):
            return False
        if isinstance(value, zoneinfo.ZoneInfo):
            return False
        if isinstance(value, Path):
            return not value.is_absolute()
        return True
    return False


def _runtime_validate_typing_alias(value: Any, *, alias_name: str) -> dict[str, Any]:
    expected_value = _typing.Mapping if alias_name == "Mapping" else _typing.Sequence
    expected_origin = (
        _collections_abc.Mapping
        if alias_name == "Mapping"
        else _collections_abc.Sequence
    )
    expected_nparams = 2 if alias_name == "Mapping" else 1
    if alias_name not in {"Mapping", "Sequence"} or value is not expected_value:
        _reject("runtime_typing_alias_invalid")
    state = vars(value)
    if (
        type(value) is not type(_typing.Mapping)
        or type(value).__module__ != "typing"
        or type(value).__qualname__ != "_SpecialGenericAlias"
        or set(state)
        != {"_inst", "_name", "__origin__", "__slots__", "_nparams", "__doc__"}
        or state["_inst"] is not True
        or state["_name"] != alias_name
        or state["_nparams"] != expected_nparams
        or state["__slots__"] is not None
        or state["__origin__"] is not expected_origin
        or type(state["__doc__"]) is not str
    ):
        _reject("runtime_typing_alias_invalid")
    return dict(state)


def _runtime_validate_weak_registry(
    value: weakref.WeakKeyDictionary[Any, Any],
) -> dict[str, Any]:
    if type(value) is not weakref.WeakKeyDictionary:
        _reject("runtime_weak_registry_invalid")
    state = vars(value)
    # Inspect the raw implementation state before ``len`` or any other
    # operation that may commit/scrub pending weak-reference removals.
    if (
        set(state)
        != {"data", "_pending_removals", "_iterating", "_dirty_len", "_remove"}
        or type(state["data"]) is not dict
        or state["data"] != {}
        or type(state["_pending_removals"]) is not list
        or state["_pending_removals"] != []
        or type(state["_iterating"]) is not set
        or state["_iterating"] != set()
        or state["_dirty_len"] is not False
    ):
        _reject("runtime_weak_registry_invalid")
    if len(value) != 0:
        _reject("runtime_weak_registry_invalid")
    callback = state["_remove"]
    if (
        type(callback) is not types.FunctionType
        or callback.__module__ != "weakref"
        or callback.__qualname__ != "WeakKeyDictionary.__init__.<locals>.remove"
        or callback.__closure__ is not None
        or callback.__kwdefaults__ is not None
        or callback.__annotations__ != {}
        or type(callback.__defaults__) is not tuple
        or len(callback.__defaults__) != 1
    ):
        _reject("runtime_weak_registry_invalid")
    self_ref = callback.__defaults__[0]
    if (
        type(self_ref) is not weakref.ReferenceType
        or self_ref() is not value
        or self_ref.__callback__ is not None
    ):
        _reject("runtime_weak_registry_invalid")
    return {"callback": callback, "self_ref": self_ref}


def _runtime_bind_relative_path_adapters(
    repo_modules: Mapping[str, types.ModuleType],
) -> dict[int, tuple[Any, tuple[tuple[str, str], ...], str]]:
    expected_bindings = {
        (module_name, global_name): posix_value
        for module_name, global_name, posix_value in _RUNTIME_RELATIVE_PATH_BINDINGS
    }
    observed_bindings: dict[tuple[str, str], Path] = {}
    for module_name, module in repo_modules.items():
        for global_name, value in vars(module).items():
            if isinstance(value, Path) and not value.is_absolute():
                observed_bindings[(module_name, global_name)] = value
    if set(observed_bindings) != set(expected_bindings):
        _reject("runtime_relative_path_allowlist_invalid")
    grouped: dict[int, tuple[Any, list[tuple[str, str]], str]] = {}
    for binding, expected_posix in expected_bindings.items():
        value = observed_bindings[binding]
        if (
            type(value) is not _pathlib.WindowsPath
            or value.as_posix() != expected_posix
            or value.drive != ""
            or value.root != ""
            or value.anchor != ""
            or not value.parts
            or any(part in {"", ".", ".."} for part in value.parts)
        ):
            _reject("runtime_relative_path_allowlist_invalid")
        prior = grouped.get(id(value))
        if prior is None:
            grouped[id(value)] = (value, [binding], expected_posix)
        else:
            if prior[0] is not value or prior[2] != expected_posix:
                _reject("runtime_relative_path_allowlist_invalid")
            prior[1].append(binding)
    store_state = observed_bindings[
        (
            "agent_benchmark.sec_gemma_online_risk_overlay_store",
            "STATE_RELATIVE_DIRECTORY",
        )
    ]
    vault_state = observed_bindings[
        (
            "agent_benchmark.sec_gemma_online_risk_overlay_vault",
            "STATE_RELATIVE_DIRECTORY",
        )
    ]
    if store_state is not vault_state or len(grouped) != 4:
        _reject("runtime_relative_path_allowlist_invalid")
    return {
        identifier: (
            value,
            tuple(
                sorted(
                    bindings,
                    key=lambda item: (
                        item[0].encode("utf-8"), item[1].encode("utf-8")
                    ),
                )
            ),
            posix_value,
        )
        for identifier, (value, bindings, posix_value) in grouped.items()
    }


def _runtime_bind_typing_alias_adapters(
    repo_modules: Mapping[str, types.ModuleType],
) -> dict[int, tuple[Any, tuple[tuple[str, str], ...], str]]:
    expected = {
        (module_name, global_name): alias_name
        for module_name, global_name, alias_name in _RUNTIME_TYPING_ALIAS_BINDINGS
    }
    special_type = type(_typing.Mapping)
    observed: dict[tuple[str, str], Any] = {}
    for module_name, module in repo_modules.items():
        for global_name, value in vars(module).items():
            if type(value) is special_type:
                observed[(module_name, global_name)] = value
    if set(observed) != set(expected):
        _reject("runtime_typing_alias_allowlist_invalid")
    grouped: dict[int, tuple[Any, list[tuple[str, str]], str]] = {}
    for binding, alias_name in expected.items():
        value = observed[binding]
        _runtime_validate_typing_alias(value, alias_name=alias_name)
        prior = grouped.get(id(value))
        if prior is None:
            grouped[id(value)] = (value, [binding], alias_name)
        else:
            if prior[0] is not value or prior[2] != alias_name:
                _reject("runtime_typing_alias_allowlist_invalid")
            prior[1].append(binding)
    if (
        len(grouped) != 2
        or observed[next(binding for binding, name in expected.items() if name == "Mapping")]
        is not _typing.Mapping
        or observed[next(binding for binding, name in expected.items() if name == "Sequence")]
        is not _typing.Sequence
        or _typing.Mapping is _typing.Sequence
    ):
        _reject("runtime_typing_alias_allowlist_invalid")
    return {
        identifier: (
            value,
            tuple(
                sorted(
                    bindings,
                    key=lambda item: (
                        item[0].encode("utf-8"), item[1].encode("utf-8")
                    ),
                )
            ),
            alias_name,
        )
        for identifier, (value, bindings, alias_name) in grouped.items()
    }


def _runtime_nested_code_objects(value: types.CodeType) -> tuple[types.CodeType, ...]:
    result: list[types.CodeType] = []
    queue = [value]
    seen: set[int] = set()
    while queue:
        current = queue.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        queue.extend(
            item for item in current.co_consts if isinstance(item, types.CodeType)
        )
    return tuple(result)


def _runtime_class_descriptors(value: type) -> list[tuple[str, str, Any]]:
    rows: list[tuple[str, str, Any]] = []
    for name, member in value.__dict__.items():
        if inspect.isfunction(member):
            rows.append((name, "method", member))
        elif isinstance(member, staticmethod):
            rows.append((name, "staticmethod", member.__func__))
        elif isinstance(member, classmethod):
            rows.append((name, "classmethod", member.__func__))
        elif isinstance(member, property):
            for suffix, function in (
                ("fget", member.fget),
                ("fset", member.fset),
                ("fdel", member.fdel),
            ):
                if function is not None:
                    rows.append((f"{name}.{suffix}", "property", function))
        elif inspect.isclass(member) and member.__module__ == value.__module__:
            rows.append((name, "nested_class", member))
    rows.sort(key=lambda row: (row[0].encode("utf-8"), row[1].encode("utf-8")))
    return rows


def _runtime_collect_namespace_roots(
    repo_modules: Mapping[str, types.ModuleType],
) -> tuple[dict[str, set[str]], list[tuple[str, str, Any]]]:
    """Reproduce the inherited namespace scan without natural-name keying."""

    names_by_module = {name: set() for name in repo_modules}
    repository_callables: dict[int, Any] = {}
    queue: list[Any] = []

    def add_repository_callable(value: Any) -> None:
        if not _runtime_is_supported_callable(value):
            return
        owner = getattr(value, "__module__", None)
        if type(owner) is not str:
            owner = type(value).__module__
        if owner not in repo_modules:
            return
        prior = repository_callables.setdefault(id(value), value)
        if prior is not value:
            _reject("runtime_owner_graph_identity_reuse")
        if prior is value and value not in queue and id(value) not in scanned:
            queue.append(value)

    scanned: set[int] = set()
    for module_name, module in repo_modules.items():
        for global_name, value in vars(module).items():
            if (
                (inspect.isfunction(value) or inspect.isclass(value))
                and getattr(value, "__module__", None) == module_name
            ):
                names_by_module[module_name].add(global_name)
                add_repository_callable(value)
    while queue:
        value = queue.pop()
        if id(value) in scanned:
            continue
        scanned.add(id(value))
        if inspect.isclass(value):
            for _name, _kind, member in _runtime_class_descriptors(value):
                add_repository_callable(member)
            continue
        if not inspect.isfunction(value):
            continue
        module_name = value.__module__
        module_namespace = vars(repo_modules[module_name])
        referenced_names: set[str] = set()
        for code in _runtime_nested_code_objects(value.__code__):
            referenced_names.update(
                name for name in code.co_names if name in module_namespace
            )
        names_by_module[module_name].update(referenced_names)
        for global_name in referenced_names:
            add_repository_callable(module_namespace[global_name])

    for module_name, global_name, _posix_value in _RUNTIME_RELATIVE_PATH_BINDINGS:
        names_by_module[module_name].add(global_name)
    for module_name, global_name, _alias_name in _RUNTIME_TYPING_ALIAS_BINDINGS:
        names_by_module[module_name].add(global_name)

    roots = [
        (module_name, global_name, vars(repo_modules[module_name])[global_name])
        for module_name in sorted(repo_modules, key=lambda value: value.encode("utf-8"))
        for global_name in sorted(
            names_by_module[module_name], key=lambda value: value.encode("utf-8")
        )
    ]
    return names_by_module, roots


def _runtime_owner_graph_edges(
    value: Any,
    *,
    repo_modules: Mapping[str, types.ModuleType],
) -> list[tuple[str, str, Any]]:
    edges: list[tuple[str, str, Any]] = []
    if inspect.isclass(value):
        if value.__module__ not in repo_modules:
            return edges
        edges.extend(
            ("class_base", str(index), base)
            for index, base in enumerate(value.__bases__)
        )
        for name, member in value.__dict__.items():
            if inspect.isfunction(member):
                edges.append(("class_method", name, member))
            elif isinstance(member, staticmethod):
                edges.append(("class_staticmethod", name, member.__func__))
            elif isinstance(member, classmethod):
                edges.append(("class_classmethod", name, member.__func__))
            elif isinstance(member, property):
                if member.fget is not None:
                    edges.append(("class_property_fget", name, member.fget))
                if member.fset is not None:
                    edges.append(("class_property_fset", name, member.fset))
                if member.fdel is not None:
                    edges.append(("class_property_fdel", name, member.fdel))
            elif inspect.isclass(member) and member.__module__ == value.__module__:
                edges.append(("class_nested_class", name, member))
        annotations = getattr(value, "__annotations__", {})
        if type(annotations) is not dict or any(
            type(name) is not str for name in annotations
        ):
            _reject("runtime_owner_graph_class_annotations_invalid")
        edges.extend(
            ("class_annotation", name, annotations[name])
            for name in sorted(annotations, key=lambda item: item.encode("utf-8"))
        )
    elif inspect.isfunction(value):
        edges.extend(
            ("function_default", str(index), item)
            for index, item in enumerate(value.__defaults__ or ())
        )
        kwdefaults = value.__kwdefaults__ or {}
        if type(kwdefaults) is not dict or any(type(name) is not str for name in kwdefaults):
            _reject("runtime_owner_graph_function_defaults_invalid")
        edges.extend(
            ("function_kwdefault", name, kwdefaults[name])
            for name in sorted(kwdefaults, key=lambda item: item.encode("utf-8"))
        )
        annotations = value.__annotations__
        if type(annotations) is not dict or any(type(name) is not str for name in annotations):
            _reject("runtime_owner_graph_function_annotations_invalid")
        edges.extend(
            ("function_annotation", name, annotations[name])
            for name in sorted(annotations, key=lambda item: item.encode("utf-8"))
        )
        closure = value.__closure__ or ()
        if len(closure) != len(value.__code__.co_freevars):
            _reject("runtime_owner_graph_closure_invalid")
        for name, cell in zip(value.__code__.co_freevars, closure, strict=True):
            try:
                child = cell.cell_contents
            except ValueError:
                _reject("loaded_code_empty_closure_cell")
            edges.append(("function_closure", name, child))
        if value.__module__ in repo_modules:
            names: set[str] = set()
            for code in _runtime_nested_code_objects(value.__code__):
                names.update(
                    name for name in code.co_names if name in value.__globals__
                )
            edges.extend(
                ("function_referenced_global", name, value.__globals__[name])
                for name in sorted(names, key=lambda item: item.encode("utf-8"))
            )
    elif type(value) is list:
        edges.extend(("list_item", str(index), item) for index, item in enumerate(value))
    elif type(value) is tuple:
        edges.extend(("tuple_item", str(index), item) for index, item in enumerate(value))
    elif type(value) is weakref.WeakKeyDictionary:
        details = _runtime_validate_weak_registry(value)
        edges.append(("weak_registry_remove", "_remove", details["callback"]))
    elif type(value) is weakref.ReferenceType:
        referent = value()
        if (
            type(referent) is not weakref.WeakKeyDictionary
            or value.__callback__ is not None
        ):
            _reject("runtime_owner_graph_weak_reference_unsupported")
        edges.append(("weak_registry_self_ref", "referent", referent))
    elif _runtime_is_typing_semantic(value):
        alias_name = getattr(value, "_name", None)
        if alias_name not in {"Mapping", "Sequence"}:
            _reject("runtime_typing_alias_invalid")
        state = _runtime_validate_typing_alias(value, alias_name=alias_name)
        edges.append(("typing_alias_type", "typing._SpecialGenericAlias", type(value)))
        edges.append(("typing_alias_origin", alias_name, state["__origin__"]))
    elif isinstance(value, Path):
        if not value.is_absolute():
            edges.append(("relative_path_type", "pathlib.WindowsPath", type(value)))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        edges.extend(
            ("dataclass_field", field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
        edges.append(("dataclass_type", "type", type(value)))
    elif isinstance(value, enum.Enum):
        edges.append(("enum_type", "type", type(value)))
    elif isinstance(value, (types.MappingProxyType, Mapping)):
        for key, child in value.items():
            try:
                owner_key = _runtime_owner_free_value(key)
            except V315PreflightError:
                _reject("runtime_owner_graph_identity_mapping_key_unsupported")
            if _runtime_value_has_graph_identity(child):
                token = hashlib.sha256(owner_key).hexdigest()
                edges.append(("mapping_value", token, child))
    elif type(value) in {set, frozenset} and any(
        _runtime_value_has_graph_identity(item) for item in value
    ):
        _reject("runtime_owner_graph_identity_set_unsupported")
    edge_tokens = [(kind, name) for kind, name, _child in edges]
    if (
        any(kind not in _RUNTIME_OWNER_EDGE_KINDS for kind, _name in edge_tokens)
        or len(edge_tokens) != len(set(edge_tokens))
        or any(type(name) is not str or not name for _kind, name in edge_tokens)
    ):
        _reject("runtime_owner_graph_edge_invalid")
    return sorted(
        edges,
        key=lambda edge: contract.canonical_json_bytes([edge[0], edge[1]]),
    )


def _runtime_build_owner_graph(
    *,
    repo_modules: Mapping[str, types.ModuleType],
    roots: Sequence[tuple[str, str, Any]],
) -> dict[str, Any]:
    """Settle one deterministic shortest owner path, then seal all aliases."""

    heap: list[tuple[int, bytes, int, Any, tuple[tuple[str, ...], ...]]] = []
    serial = 0
    held: dict[int, Any] = {}
    settled: dict[int, Any] = {}
    paths: dict[int, tuple[tuple[str, ...], ...]] = {}
    incoming: dict[int, set[tuple[str, ...]]] = {}

    def hold(value: Any) -> int:
        identifier = id(value)
        prior = held.setdefault(identifier, value)
        if prior is not value:
            _reject("runtime_owner_graph_identity_reuse")
        return identifier

    def push(value: Any, path: tuple[tuple[str, ...], ...]) -> None:
        nonlocal serial
        hold(value)
        path_bytes = contract.canonical_json_bytes([list(item) for item in path])
        heapq.heappush(heap, (len(path), path_bytes, serial, value, path))
        serial += 1

    sorted_roots = sorted(
        roots,
        key=lambda root: contract.canonical_json_bytes(
            [["module_global", root[0], root[1]]]
        ),
    )
    if len({(module, name) for module, name, _value in sorted_roots}) != len(
        sorted_roots
    ):
        _reject("runtime_owner_graph_root_duplicate")
    for module_name, global_name, value in sorted_roots:
        identifier = hold(value)
        incoming.setdefault(identifier, set()).add(
            ("root", module_name, "module_global", global_name)
        )
        push(value, (("module_global", module_name, global_name),))

    while heap:
        _length, _path_bytes, _serial, value, path = heapq.heappop(heap)
        identifier = hold(value)
        prior = settled.get(identifier)
        if prior is not None:
            if prior is not value:
                _reject("runtime_owner_graph_identity_reuse")
            continue
        settled[identifier] = value
        paths[identifier] = path
        holder_hash = hashlib.sha256(
            contract.canonical_json_bytes([list(item) for item in path])
        ).hexdigest()
        for edge_kind, edge_name, child in _runtime_owner_graph_edges(
            value, repo_modules=repo_modules
        ):
            child_identifier = hold(child)
            incoming.setdefault(child_identifier, set()).add(
                ("edge", holder_hash, edge_kind, edge_name)
            )
            push(child, (*path, (edge_kind, edge_name)))

    if set(held) != set(settled) or set(settled) != set(incoming):
        _reject("runtime_owner_graph_unrooted_identity")
    owner_slots: dict[int, tuple[Any, tuple[tuple[str, ...], ...]]] = {}
    owner_slot_hashes: dict[int, tuple[Any, str]] = {}
    for identifier, value in settled.items():
        slots = tuple(
            sorted(
                incoming[identifier],
                key=lambda slot: contract.canonical_json_bytes(list(slot)),
            )
        )
        if not slots or any("discovered:" in item for slot in slots for item in slot):
            _reject("runtime_owner_graph_slot_invalid")
        digest = contract.canonical_sha256([list(slot) for slot in slots])
        owner_slots[identifier] = (value, slots)
        owner_slot_hashes[identifier] = (value, digest)
    return {
        "strong_objects": settled,
        "canonical_paths": paths,
        "owner_slots": owner_slots,
        "owner_slot_hashes": owner_slot_hashes,
    }


def _runtime_owner_path_sha256(graph: Mapping[str, Any], value: Any) -> str:
    entry = graph["canonical_paths"].get(id(value))
    strong = graph["strong_objects"].get(id(value))
    if entry is None or strong is not value:
        _reject("runtime_owner_graph_unrooted_identity")
    return hashlib.sha256(
        contract.canonical_json_bytes([list(item) for item in entry])
    ).hexdigest()


def _runtime_validate_closure_state_allowlist(
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    module_name = "agent_benchmark.sec_filing_gemma_market_acquirer"
    prefix = "_build_owned_transport_capability_boundary.<locals>."
    holder_names = {
        "issue": f"{prefix}issue",
        "unwrap": f"{prefix}unwrap",
        "bind_to_claim": f"{prefix}bind_to_claim",
        "consume": f"{prefix}consume",
        "OwnedDevelopmentMarketAcquisition.__init__": (
            f"{prefix}OwnedDevelopmentMarketAcquisition.__init__"
        ),
        "OwnedMarketTransportCapability.__init__": (
            f"{prefix}OwnedMarketTransportCapability.__init__"
        ),
    }
    functions: dict[str, types.FunctionType] = {}
    for value in graph["strong_objects"].values():
        if type(value) is types.FunctionType and value.__module__ == module_name:
            if value.__qualname__ in holder_names.values():
                prior = functions.setdefault(value.__qualname__, value)
                if prior is not value:
                    _reject("runtime_closure_state_holder_duplicate")
    if set(functions) != set(holder_names.values()):
        _reject("runtime_closure_state_holder_missing")

    closure_values: dict[tuple[str, str], Any] = {}
    for short_name, qualname in holder_names.items():
        function = functions[qualname]
        closure = function.__closure__ or ()
        if len(closure) != len(function.__code__.co_freevars):
            _reject("runtime_closure_state_invalid")
        for free_name, cell in zip(
            function.__code__.co_freevars, closure, strict=True
        ):
            try:
                closure_values[(short_name, free_name)] = cell.cell_contents
            except ValueError:
                _reject("runtime_closure_state_invalid")

    expected_owners = {
        "issuer": (
            ("issue", "issuer"),
            ("OwnedDevelopmentMarketAcquisition.__init__", "issuer"),
            ("OwnedMarketTransportCapability.__init__", "issuer"),
        ),
        "missing": (
            ("unwrap", "missing"),
            ("bind_to_claim", "missing"),
            ("consume", "missing"),
        ),
        "acquisition_results": (
            ("issue", "acquisition_results"),
            ("unwrap", "acquisition_results"),
        ),
        "capability_bindings": (
            ("issue", "capability_bindings"),
            ("bind_to_claim", "capability_bindings"),
            ("consume", "capability_bindings"),
        ),
        "registry_lock": (
            ("issue", "registry_lock"),
            ("unwrap", "registry_lock"),
            ("bind_to_claim", "registry_lock"),
            ("consume", "registry_lock"),
        ),
    }
    state: dict[str, Any] = {}
    for label, owners in expected_owners.items():
        values = [closure_values.get(owner) for owner in owners]
        if any(value is None for value in values) or any(
            value is not values[0] for value in values[1:]
        ):
            _reject("runtime_closure_state_owner_invalid")
        state[label] = values[0]
    if len({id(value) for value in state.values()}) != 5:
        _reject("runtime_closure_state_identity_invalid")
    if (
        type(state["issuer"]) is not object
        or type(state["missing"]) is not object
        or type(state["acquisition_results"]) is not weakref.WeakKeyDictionary
        or type(state["capability_bindings"]) is not weakref.WeakKeyDictionary
        or type(state["registry_lock"]) is not _RUNTIME_LOCK_TYPE
        or state["registry_lock"].locked() is not False
    ):
        _reject("runtime_closure_state_type_invalid")

    expected_pairs = {
        owner: state[label]
        for label, owners in expected_owners.items()
        for owner in owners
    }
    observed_stateful_pairs: dict[tuple[str, str], Any] = {}
    for value in graph["strong_objects"].values():
        if type(value) is not types.FunctionType or value.__closure__ is None:
            continue
        short_name = next(
            (
                short
                for short, qualname in holder_names.items()
                if value.__module__ == module_name and value.__qualname__ == qualname
            ),
            None,
        )
        for free_name, cell in zip(
            value.__code__.co_freevars, value.__closure__, strict=True
        ):
            try:
                child = cell.cell_contents
            except ValueError:
                _reject("runtime_closure_state_invalid")
            if type(child) in {
                object,
                weakref.WeakKeyDictionary,
                _RUNTIME_LOCK_TYPE,
                _RUNTIME_RLOCK_TYPE,
            }:
                if short_name is None:
                    _reject("runtime_closure_state_extra")
                observed_stateful_pairs[(short_name, free_name)] = child
    if set(observed_stateful_pairs) != set(expected_pairs) or any(
        observed_stateful_pairs[pair] is not expected
        for pair, expected in expected_pairs.items()
    ):
        _reject("runtime_closure_state_extra")

    for label, owners in expected_owners.items():
        value = state[label]
        slot_entry = graph["owner_slots"].get(id(value))
        if slot_entry is None or slot_entry[0] is not value:
            _reject("runtime_closure_state_owner_slots_missing")
        observed_closure_slots = {
            slot for slot in slot_entry[1]
            if slot[0] == "edge" and slot[2] == "function_closure"
        }
        expected_closure_slots = {
            (
                "edge",
                _runtime_owner_path_sha256(graph, functions[holder_names[short]]),
                "function_closure",
                free_name,
            )
            for short, free_name in owners
        }
        if observed_closure_slots != expected_closure_slots:
            _reject("runtime_closure_state_owner_slots_invalid")
        allowed_edge_kinds = (
            {"function_closure", "weak_registry_self_ref"}
            if label in {"acquisition_results", "capability_bindings"}
            else {"function_closure"}
        )
        if any(
            slot[0] != "edge" or slot[2] not in allowed_edge_kinds
            for slot in slot_entry[1]
        ) or (
            label in {"acquisition_results", "capability_bindings"}
            and len(
                [
                    slot
                    for slot in slot_entry[1]
                    if slot[2] == "weak_registry_self_ref"
                ]
            )
            != 1
        ):
            _reject("runtime_closure_state_owner_slots_invalid")

    registry_details = {
        label: _runtime_validate_weak_registry(state[label])
        for label in ("acquisition_results", "capability_bindings")
    }
    if (
        registry_details["acquisition_results"]["callback"]
        is registry_details["capability_bindings"]["callback"]
        or registry_details["acquisition_results"]["self_ref"]
        is registry_details["capability_bindings"]["self_ref"]
    ):
        _reject("runtime_weak_registry_identity_invalid")
    state["registry_details"] = registry_details
    state["holder_functions"] = functions
    return state


def _runtime_component_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_callable_owner_material(
    value: Any,
    *,
    context: _RuntimeFingerprintContext,
) -> tuple[str | None, str | None]:
    owner_module = getattr(value, "__module__", None)
    if type(owner_module) is not str:
        owner_module = type(value).__module__
    repo = context.repo_rows.get(owner_module)
    if repo is not None:
        return repo["literal_sha256"], None
    file_row = context.dependency_files.get(owner_module)
    if file_row is None:
        return None, context.python_dll_sha256
    if file_row["origin_kind"] == "extension_module":
        binary = context.binary_by_path.get(file_row["resolved_path_sha256"])
        if binary is None or binary["literal_sha256"] != file_row["literal_sha256"]:
            _reject("runtime_fingerprint_native_binary_missing")
        return None, binary["literal_sha256"]
    return file_row["literal_sha256"], None


def _runtime_function_referenced_globals(
    function: types.FunctionType,
    *,
    context: _RuntimeFingerprintContext,
) -> bytes:
    names: set[str] = set()
    for code in _runtime_nested_code_objects(function.__code__):
        names.update(name for name in code.co_names if name in function.__globals__)
    fields: list[bytes] = []
    for name in sorted(names, key=lambda value: value.encode("utf-8")):
        fields.append(
            context._pack(
                "referenced_global",
                (context._text(name), context.value(function.__globals__[name])),
            )
        )
    return context._pack("referenced_globals", tuple(fields))


def _runtime_function_row(
    function: types.FunctionType,
    *,
    context: _RuntimeFingerprintContext,
) -> dict[str, Any]:
    owner_module = function.__module__
    owner_file, owning_binary = _runtime_callable_owner_material(
        function, context=context
    )
    if owning_binary is not None:
        _reject("runtime_fingerprint_python_function_binary_invalid")
    owner_slots = context.owner_slot_hashes.get(id(function))
    if owner_slots is None or owner_slots[0] is not function:
        _reject("runtime_fingerprint_callable_owner_slots_missing")
    try:
        closure_values = tuple(
            cell.cell_contents for cell in (function.__closure__ or ())
        )
    except ValueError:
        _reject("runtime_fingerprint_empty_closure")
    return {
        "qualified_name": context.callable_qualified_name(function),
        "kind": "function",
        "owner_module": owner_module,
        "owner_file_sha256": owner_file,
        "object_type": f"{type(function).__module__}.{type(function).__qualname__}",
        "code_sha256": _runtime_component_sha256(
            context.code(function.__code__, owner_module=owner_module)
        ),
        "defaults_sha256": _runtime_component_sha256(
            context.value(function.__defaults__)
        ),
        "kwdefaults_sha256": _runtime_component_sha256(
            context.value(function.__kwdefaults__)
        ),
        "annotations_sha256": _runtime_component_sha256(
            context.value(function.__annotations__)
        ),
        "closure_sha256": _runtime_component_sha256(
            context.value(closure_values)
        ),
        "referenced_globals_sha256": _runtime_component_sha256(
            _runtime_function_referenced_globals(function, context=context)
        ),
        "descriptor_members_sha256": None,
        "owning_binary_sha256": None,
        "owner_slots_sha256": owner_slots[1],
    }


def _runtime_class_row(
    value: type,
    *,
    context: _RuntimeFingerprintContext,
) -> dict[str, Any]:
    owner_module = value.__module__
    owner_file, owning_binary = _runtime_callable_owner_material(value, context=context)
    owner_slots = context.owner_slot_hashes.get(id(value))
    if owner_slots is None or owner_slots[0] is not value:
        _reject("runtime_fingerprint_callable_owner_slots_missing")
    descriptors: list[bytes] = [
        context._pack("base", (context.callable_reference(base),))
        for base in value.__bases__
    ]
    for name, kind, member in _runtime_class_descriptors(value):
        descriptors.append(
            context._pack(
                "descriptor",
                (
                    context._text(name),
                    context._text(kind),
                    context.callable_reference(member),
                ),
            )
        )
    return {
        "qualified_name": context.callable_qualified_name(value),
        "kind": "class",
        "owner_module": owner_module,
        "owner_file_sha256": owner_file,
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "code_sha256": None,
        "defaults_sha256": None,
        "kwdefaults_sha256": None,
        "annotations_sha256": _runtime_component_sha256(
            context.value(getattr(value, "__annotations__", {}))
        ),
        "closure_sha256": None,
        "referenced_globals_sha256": None,
        "descriptor_members_sha256": _runtime_component_sha256(
            context._pack("class_descriptors", tuple(descriptors))
        ),
        "owning_binary_sha256": owning_binary,
        "owner_slots_sha256": owner_slots[1],
    }


def _runtime_external_callable_row(
    value: Any,
    *,
    context: _RuntimeFingerprintContext,
) -> dict[str, Any]:
    if inspect.isfunction(value):
        # Bind its own code and values, but deliberately do not traverse its
        # external global graph.
        owner_module = value.__module__
        owner_file, owning_binary = _runtime_callable_owner_material(
            value, context=context
        )
        owner_slots = context.owner_slot_hashes.get(id(value))
        if owner_slots is None or owner_slots[0] is not value:
            _reject("runtime_fingerprint_callable_owner_slots_missing")
        try:
            closure = tuple(cell.cell_contents for cell in (value.__closure__ or ()))
        except ValueError:
            _reject("runtime_fingerprint_empty_closure")
        return {
            "qualified_name": context.callable_qualified_name(value),
            "kind": "function",
            "owner_module": owner_module,
            "owner_file_sha256": owner_file,
            "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "code_sha256": _runtime_component_sha256(
                context.code(value.__code__, owner_module=owner_module)
            ),
            "defaults_sha256": _runtime_component_sha256(context.value(value.__defaults__)),
            "kwdefaults_sha256": _runtime_component_sha256(context.value(value.__kwdefaults__)),
            "annotations_sha256": _runtime_component_sha256(context.value(value.__annotations__)),
            "closure_sha256": _runtime_component_sha256(context.value(closure)),
            "referenced_globals_sha256": None,
            "descriptor_members_sha256": None,
            "owning_binary_sha256": owning_binary,
            "owner_slots_sha256": owner_slots[1],
        }
    if inspect.isclass(value):
        owner_module = value.__module__
        owner_file, owning_binary = _runtime_callable_owner_material(
            value, context=context
        )
        owner_slots = context.owner_slot_hashes.get(id(value))
        if owner_slots is None or owner_slots[0] is not value:
            _reject("runtime_fingerprint_callable_owner_slots_missing")
        return {
            "qualified_name": context.callable_qualified_name(value),
            "kind": "class",
            "owner_module": owner_module,
            "owner_file_sha256": owner_file,
            "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "code_sha256": None,
            "defaults_sha256": None,
            "kwdefaults_sha256": None,
            "annotations_sha256": None,
            "closure_sha256": None,
            "referenced_globals_sha256": None,
            "descriptor_members_sha256": _runtime_component_sha256(
                context._pack(
                    "external_class",
                    (
                        context.callable_reference(value),
                        context._text(f"{type(value).__module__}.{type(value).__qualname__}"),
                    ),
                )
            ),
            "owning_binary_sha256": owning_binary,
            "owner_slots_sha256": owner_slots[1],
        }
    owner_module = getattr(value, "__module__", None)
    qualname = context.callable_qualified_name(value)
    if type(owner_module) is not str:
        owner_module = type(value).__module__
    owner_file, owning_binary = _runtime_callable_owner_material(value, context=context)
    owner_slots = context.owner_slot_hashes.get(id(value))
    if owner_slots is None or owner_slots[0] is not value:
        _reject("runtime_fingerprint_callable_owner_slots_missing")
    if owning_binary is None:
        # A Python callable object is bound to the byte identity of its class
        # definition.  Native/builtin callables bind the owning image.
        owning_binary = None
    return {
        "qualified_name": qualname,
        "kind": "native_callable",
        "owner_module": owner_module,
        "owner_file_sha256": owner_file,
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "code_sha256": None,
        "defaults_sha256": None,
        "kwdefaults_sha256": None,
        "annotations_sha256": None,
        "closure_sha256": None,
        "referenced_globals_sha256": None,
        "descriptor_members_sha256": None,
        "owning_binary_sha256": owning_binary,
        "owner_slots_sha256": owner_slots[1],
    }


def _build_loaded_code_manifest(
    *,
    checkpoint_name: str,
    repository_manifest: Mapping[str, Any],
    dependency_manifest: Mapping[str, Any],
    critical_objects: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint_name not in RUNTIME_CHECKPOINT_NAMES:
        _reject("loaded_code_checkpoint_invalid")
    if (
        len(threading.enumerate()) != 1
        or threading.current_thread() is not threading.main_thread()
    ):
        _reject("loaded_code_thread_state_invalid")
    repository = validate_repository_runtime_manifest(repository_manifest)
    dependency = validate_execution_dependency_manifest(dependency_manifest)
    context = _RuntimeFingerprintContext(
        repository=repository, dependency=dependency
    )
    repo_modules: dict[str, types.ModuleType] = {}
    for row in repository["modules"]:
        module = sys.modules.get(row["module_name"])
        if type(module) is not types.ModuleType:
            _reject("loaded_code_repository_module_missing")
        repo_modules[row["module_name"]] = module
    pseudo_alias_objects: dict[str, Any] = {}
    for row in dependency["normalized_pseudo_module_aliases"]:
        owner = sys.modules.get(row["owner_module"])
        if (
            row["alias_name"] in sys.modules
            or type(owner) is not types.ModuleType
            or not hasattr(owner, row["owner_attribute"])
        ):
            _reject("runtime_pseudo_module_alias_invalid")
        value = getattr(owner, row["owner_attribute"])
        if f"{type(value).__module__}.{type(value).__qualname__}" != row["object_type"]:
            _reject("runtime_pseudo_module_alias_invalid")
        pseudo_alias_objects[row["alias_name"]] = value

    relative_paths = _runtime_bind_relative_path_adapters(repo_modules)
    typing_aliases = _runtime_bind_typing_alias_adapters(repo_modules)
    names_by_module, roots = _runtime_collect_namespace_roots(repo_modules)
    graph = _runtime_build_owner_graph(repo_modules=repo_modules, roots=roots)
    context.owner_slots = dict(graph["owner_slots"])
    context.owner_slot_hashes = dict(graph["owner_slot_hashes"])
    context.relative_path_bindings = dict(relative_paths)
    context.typing_alias_bindings = dict(typing_aliases)

    closure_state = _runtime_validate_closure_state_allowlist(graph)
    closure_objects = {
        label: closure_state[label]
        for label in (
            "issuer",
            "missing",
            "acquisition_results",
            "capability_bindings",
            "registry_lock",
        )
    }
    closure_ids = {id(value) for value in closure_objects.values()}
    stateful_identity: dict[str, Any] = dict(closure_objects)
    for identifier, value in graph["strong_objects"].items():
        if type(value) not in {
            object,
            weakref.WeakKeyDictionary,
            _RUNTIME_LOCK_TYPE,
            _RUNTIME_RLOCK_TYPE,
        }:
            continue
        slot_entry = graph["owner_slots"][identifier]
        owner_hash = graph["owner_slot_hashes"][identifier][1]
        root_slots = [slot for slot in slot_entry[1] if slot[0] == "root"]
        if identifier in closure_ids:
            if root_slots:
                _reject("runtime_closure_state_cross_module_alias")
        elif type(value) is weakref.WeakKeyDictionary:
            _reject("runtime_weak_registry_extra")
        elif len(root_slots) != 1:
            _reject("loaded_code_stateful_root_invalid")
        else:
            label = f"{root_slots[0][1]}.{root_slots[0][3]}"
            stateful_identity[label] = value
        if type(value) is object:
            context.identity_sentinels[identifier] = (value, owner_hash)
        elif type(value) in {_RUNTIME_LOCK_TYPE, _RUNTIME_RLOCK_TYPE}:
            context.lock_owners[identifier] = (value, owner_hash)
            context._lock(value)
        else:
            context.weak_registry_owners[identifier] = (value, owner_hash)

    weak_callback_identity: dict[str, Any] = {}
    weak_self_ref_identity: dict[str, Any] = {}
    for label, details in closure_state["registry_details"].items():
        registry = closure_state[label]
        callback = details["callback"]
        self_ref = details["self_ref"]
        context.weak_self_refs[id(self_ref)] = (self_ref, registry)
        weak_callback_identity[label] = callback
        weak_self_ref_identity[label] = self_ref

    callable_objects = {
        identifier: value
        for identifier, value in graph["strong_objects"].items()
        if _runtime_is_supported_callable(value)
    }
    callable_by_key: dict[tuple[str, str, str, str], Any] = {}
    for identifier, value in callable_objects.items():
        owner_module = getattr(value, "__module__", None)
        if type(owner_module) is not str:
            owner_module = type(value).__module__
        owner_slots = context.owner_slot_hashes.get(identifier)
        if owner_slots is None or owner_slots[0] is not value:
            _reject("runtime_fingerprint_callable_owner_slots_missing")
        key = (
            owner_module,
            context.callable_qualified_name(value),
            _runtime_callable_kind(value),
            owner_slots[1],
        )
        prior = callable_by_key.setdefault(key, value)
        if prior is not value:
            _reject("loaded_code_callable_duplicate")

    namespace_rows: dict[str, list[dict[str, Any]]] = {}
    namespace_identity: dict[tuple[str, str], Any] = {}
    for module_name, module in repo_modules.items():
        entries: list[dict[str, Any]] = []
        for name in sorted(
            names_by_module[module_name], key=lambda value: value.encode("utf-8")
        ):
            value = vars(module)[name]
            namespace_identity[(module_name, name)] = value
            if isinstance(value, types.ModuleType):
                kind = "module"
            elif _runtime_is_supported_callable(value):
                owner = getattr(value, "__module__", type(value).__module__)
                kind = (
                    "repository_callable"
                    if owner in repo_modules
                    else "external_callable"
                )
            elif _runtime_is_typing_semantic(value):
                kind = "semantic_value"
            elif callable(value):
                _reject("loaded_code_stateful_callable_unsupported")
            elif type(value) is object:
                kind = "identity_sentinel"
            else:
                kind = "semantic_value"
            entries.append(
                {
                    "global_name": name,
                    "binding_kind": kind,
                    "binding_sha256": _runtime_component_sha256(
                        context.value(value)
                    ),
                }
            )
        namespace_rows[module_name] = entries

    callable_rows: list[dict[str, Any]] = []
    for key, value in sorted(
        callable_by_key.items(),
        key=lambda item: tuple(field.encode("utf-8") for field in item[0]),
    ):
        owner_module = key[0]
        row = (
            _runtime_function_row(value, context=context)
            if owner_module in repo_modules and inspect.isfunction(value)
            else _runtime_class_row(value, context=context)
            if owner_module in repo_modules and inspect.isclass(value)
            else _runtime_external_callable_row(value, context=context)
        )
        if (
            row["owner_module"],
            row["qualified_name"],
            row["kind"],
            row["owner_slots_sha256"],
        ) != key:
            _reject("loaded_code_callable_key_invalid")
        callable_rows.append(row)
    if set(context.referenced_callables) - set(callable_objects):
        _reject("runtime_fingerprint_unrooted_callable")
    module_rows = [
        {
            "module_name": row["module_name"],
            "origin_kind": "repository_source",
            "origin_identity_sha256": context.repo_origins[row["module_name"]],
            "namespace_sha256": contract.canonical_sha256(
                namespace_rows[row["module_name"]]
            ),
        }
        for row in sorted(
            repository["modules"], key=lambda item: item["module_name"].encode("utf-8")
        )
    ]
    body = {
        "schema_version": LOADED_CODE_MANIFEST_SCHEMA_VERSION,
        "checkpoint_name": checkpoint_name,
        "repository_runtime_manifest_sha256": repository[
            "repository_runtime_manifest_sha256"
        ],
        "execution_dependency_manifest_sha256": dependency[
            "execution_dependency_manifest_sha256"
        ],
        "module_count": len(module_rows),
        "module_rows": module_rows,
        "callable_count": len(callable_rows),
        "callable_rows": callable_rows,
    }
    manifest = {
        **body,
        "loaded_code_manifest_sha256": contract.canonical_sha256(body),
    }
    baseline = {
        "module_identity": dict(repo_modules),
        "namespace_identity": namespace_identity,
        "callable_identity": callable_by_key,
        "critical_identity": dict(critical_objects),
        "graph_identity": {
            _runtime_owner_path_sha256(graph, value): value
            for value in graph["strong_objects"].values()
        },
        "owner_slot_arrays": {
            _runtime_owner_path_sha256(graph, value): graph["owner_slots"][identifier][1]
            for identifier, value in graph["strong_objects"].items()
        },
        "stateful_identity": stateful_identity,
        "weak_callback_identity": weak_callback_identity,
        "weak_self_ref_identity": weak_self_ref_identity,
        "relative_path_identity": {
            bindings[0][0] + "." + bindings[0][1]: value
            for value, bindings, _posix in relative_paths.values()
        },
        "typing_alias_identity": {
            alias_name: value
            for value, _bindings, alias_name in typing_aliases.values()
        },
        "pseudo_alias_identity": pseudo_alias_objects,
        "module_names": tuple(sorted(sys.modules)),
        "binary_tokens": tuple(
            private_windows_path_sha256(path)
            for path in _runtime_loaded_image_paths()
        ),
    }
    return manifest, baseline


def validate_loaded_code_manifest(value: Any) -> dict[str, Any]:
    code = "loaded_code_manifest_invalid"
    item = _strict_mapping(value, _LOADED_CODE_FIELDS, code=code)
    _validate_self_hash(item, "loaded_code_manifest_sha256", code=code)
    if (
        item["schema_version"] != LOADED_CODE_MANIFEST_SCHEMA_VERSION
        or item["checkpoint_name"] not in RUNTIME_CHECKPOINT_NAMES
        or not _is_sha256(item["repository_runtime_manifest_sha256"])
        or not _is_sha256(item["execution_dependency_manifest_sha256"])
        or item["module_count"] != 44
        or type(item["module_rows"]) is not list
        or len(item["module_rows"]) != 44
        or type(item["callable_count"]) is not int
        or item["callable_count"] < 1
        or type(item["callable_rows"]) is not list
        or len(item["callable_rows"]) != item["callable_count"]
    ):
        _reject(code)
    modules = [
        _strict_mapping(row, _LOADED_MODULE_ROW_FIELDS, code=code)
        for row in item["module_rows"]
    ]
    callables = [
        _strict_mapping(row, _LOADED_CALLABLE_ROW_FIELDS, code=code)
        for row in item["callable_rows"]
    ]
    expected_names = sorted(
        (
            path[: -len("/__init__.py")].replace("/", ".")
            if path.endswith("/__init__.py")
            else path[:-3].replace("/", ".")
        )
        for path in (
            *contract.IMPLEMENTATION_PRODUCTION_PATHS,
            *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
            contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
        )
    )
    if any(
        type(row["module_name"]) is not str
        or not row["module_name"]
        or row["origin_kind"] != "repository_source"
        or not _is_sha256(row["origin_identity_sha256"])
        or not _is_sha256(row["namespace_sha256"])
        for row in modules
    ) or any(
        row["kind"] not in {"function", "class", "native_callable"}
        or type(row["qualified_name"]) is not str
        or not row["qualified_name"]
        or "discovered:" in row["qualified_name"]
        or "[slots-sha256:" in row["qualified_name"]
        or type(row["owner_module"]) is not str
        or not row["owner_module"]
        or type(row["object_type"]) is not str
        or not row["object_type"]
        or any(
            child is not None and not _is_sha256(child)
            for child in (
                row["owner_file_sha256"],
                row["code_sha256"],
                row["defaults_sha256"],
                row["kwdefaults_sha256"],
                row["annotations_sha256"],
                row["closure_sha256"],
                row["referenced_globals_sha256"],
                row["descriptor_members_sha256"],
                row["owning_binary_sha256"],
            )
        )
        or not _is_sha256(row["owner_slots_sha256"])
        for row in callables
    ):
        _reject(code)
    expected_module_order = sorted(
        modules, key=lambda row: row["module_name"].encode("utf-8")
    )
    expected_callable_order = sorted(
        callables,
        key=lambda row: (
            row["owner_module"].encode("utf-8"),
            row["qualified_name"].encode("utf-8"),
            row["kind"].encode("utf-8"),
            row["owner_slots_sha256"].encode("utf-8"),
        ),
    )
    module_names = [row["module_name"] for row in modules]
    callable_keys = [
        (
            row["owner_module"],
            row["qualified_name"],
            row["kind"],
            row["owner_slots_sha256"],
        )
        for row in callables
    ]
    repo_names = set(expected_names)
    if (
        modules != expected_module_order
        or module_names != expected_names
        or len(module_names) != len(set(module_names))
        or callables != expected_callable_order
        or len(callable_keys) != len(set(callable_keys))
    ):
        _reject(code)
    for row in callables:
        components = {
            name: row[name]
            for name in (
                "code_sha256", "defaults_sha256", "kwdefaults_sha256",
                "annotations_sha256", "closure_sha256",
                "referenced_globals_sha256", "descriptor_members_sha256",
            )
        }
        if (row["owner_file_sha256"] is None) == (
            row["owning_binary_sha256"] is None
        ):
            _reject(code)
        if row["kind"] == "function":
            if any(
                components[name] is None
                for name in (
                    "code_sha256", "defaults_sha256", "kwdefaults_sha256",
                    "annotations_sha256", "closure_sha256",
                )
            ) or components["descriptor_members_sha256"] is not None:
                _reject(code)
            if (
                row["owner_module"] in repo_names
                and components["referenced_globals_sha256"] is None
            ) or (
                row["owner_module"] not in repo_names
                and components["referenced_globals_sha256"] is not None
            ):
                _reject(code)
        elif row["kind"] == "class":
            if any(
                components[name] is not None
                for name in (
                    "code_sha256", "defaults_sha256", "kwdefaults_sha256",
                    "closure_sha256", "referenced_globals_sha256",
                )
            ) or components["descriptor_members_sha256"] is None:
                _reject(code)
            if (
                row["owner_module"] in repo_names
                and components["annotations_sha256"] is None
            ) or (
                row["owner_module"] not in repo_names
                and components["annotations_sha256"] is not None
            ):
                _reject(code)
        elif any(value is not None for value in components.values()):
            _reject(code)
    return {**item, "module_rows": modules, "callable_rows": callables}


def _runtime_require_deadline() -> None:
    deadline = _runtime_active_preflight_context().get("deadline_monotonic")
    if deadline is not None:
        _qualification_require_before_deadline(deadline)


def _runtime_store_bundle(
    root: Path,
    *,
    repository: Mapping[str, Any],
    dependency: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> None:
    """Write one exclusive fsynced bundle with authority as the final marker."""

    code = "runtime_authority_bundle_unwritable"
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    runtime_root = private_root / PRIVATE_RUNTIME_DIRECTORY
    loaded_root = runtime_root / PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY
    if _optional_lstat(runtime_root, code=code) is not None:
        _reject(code)
    try:
        runtime_root.mkdir(exist_ok=False)
        _fsync_directory(private_root)
        loaded_root.mkdir(exist_ok=False)
        _fsync_directory(runtime_root)
    except OSError:
        _reject(code)
    writes = [
        (
            runtime_root / PRIVATE_RUNTIME_REPOSITORY_FILENAME,
            contract.canonical_json_bytes(repository),
        ),
        (
            runtime_root / PRIVATE_RUNTIME_DEPENDENCY_FILENAME,
            contract.canonical_json_bytes(dependency),
        ),
    ]
    writes.extend(
        (
            loaded_root / f"{name}.json",
            contract.canonical_json_bytes(loaded[name]),
        )
        for name in RUNTIME_CHECKPOINT_NAMES
    )
    for path, payload in writes:
        _runtime_require_deadline()
        _qualification_write_new(path, payload, code=code)
        _runtime_require_deadline()
    _fsync_directory(loaded_root)
    _runtime_require_deadline()
    _qualification_write_new(
        runtime_root / PRIVATE_RUNTIME_AUTHORITY_FILENAME,
        contract.canonical_json_bytes(authority),
        code=code,
    )
    _fsync_directory(runtime_root)
    _runtime_require_deadline()


def create_preflight_runtime_authority(
    root: Path,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    """Create authority locally from measured I315 runtime state only."""

    repo_root = _canonical_repo_root(root)
    snapshot = validate_repository_snapshot(repository)
    ancestry = snapshot["ancestry"]
    if (
        ancestry["branch"] != contract.BRANCH_NAME
        or ancestry["parent"] != contract.PREREGISTRATION_COMMIT
        or ancestry["clean_worktree"] is not True
    ):
        _reject("runtime_authority_creation_repository_invalid")
    _runtime_require_deadline()
    repository_manifest = validate_repository_runtime_manifest(
        build_repository_runtime_manifest(
            repo_root,
            implementation_commit=ancestry["commit"],
            implementation_tree=ancestry["tree"],
            origin_url=CREATION_PRETRUST_EXPECTED_ORIGIN_URL,
        )
    )
    preload = _runtime_preload_repository_modules(repo_root)
    dependency_manifest = validate_execution_dependency_manifest(
        _build_execution_dependency_manifest(
            repo_root,
            repository_manifest=repository_manifest,
            preload=preload,
        )
    )
    first, baseline = _build_loaded_code_manifest(
        checkpoint_name=RUNTIME_CHECKPOINT_NAMES[0],
        repository_manifest=repository_manifest,
        dependency_manifest=dependency_manifest,
        critical_objects=_runtime_live_critical_objects(),
    )
    first = validate_loaded_code_manifest(first)
    loaded: dict[str, dict[str, Any]] = {RUNTIME_CHECKPOINT_NAMES[0]: first}
    for checkpoint in RUNTIME_CHECKPOINT_NAMES[1:]:
        body = {
            key: copy.deepcopy(value)
            for key, value in first.items()
            if key != "loaded_code_manifest_sha256"
        }
        body["checkpoint_name"] = checkpoint
        loaded[checkpoint] = validate_loaded_code_manifest(
            {
                **body,
                "loaded_code_manifest_sha256": contract.canonical_sha256(body),
            }
        )
    authority = validate_runtime_binding_authority(
        build_runtime_binding_authority(
            repository_runtime_manifest_sha256=repository_manifest[
                "repository_runtime_manifest_sha256"
            ],
            execution_dependency_manifest_sha256=dependency_manifest[
                "execution_dependency_manifest_sha256"
            ],
            checkpoint_loaded_code_manifest_sha256s={
                name: loaded[name]["loaded_code_manifest_sha256"]
                for name in RUNTIME_CHECKPOINT_NAMES
            },
            process_environment_sha256=dependency_manifest["python_runtime"][
                "process_environment_sha256"
            ],
            final_sys_path_tokens=_runtime_final_sys_path(repo_root)[
                "final_sys_path_tokens"
            ],
        )
    )
    if (
        authority["final_sys_path_sha256"]
        != dependency_manifest["sys_path_sha256"]
    ):
        _reject("runtime_authority_creation_path_mismatch")
    _runtime_store_bundle(
        repo_root,
        repository=repository_manifest,
        dependency=dependency_manifest,
        loaded=loaded,
        authority=authority,
    )
    _runtime_binding_sessions()[private_windows_path_sha256(repo_root)] = {
        "mode": "creation",
        "authority_sha256": authority["runtime_binding_authority_sha256"],
        "baseline": baseline,
    }
    return authority


def build_runtime_binding_authority(
    *,
    repository_runtime_manifest_sha256: str,
    execution_dependency_manifest_sha256: str,
    checkpoint_loaded_code_manifest_sha256s: Mapping[str, str],
    process_environment_sha256: str,
    final_sys_path_tokens: Sequence[str],
) -> dict[str, Any]:
    checkpoints = dict(checkpoint_loaded_code_manifest_sha256s)
    if (
        set(checkpoints) != set(RUNTIME_CHECKPOINT_NAMES)
        or any(not _is_sha256(value) for value in checkpoints.values())
        or not isinstance(final_sys_path_tokens, Sequence)
        or isinstance(final_sys_path_tokens, (str, bytes, bytearray))
        or not final_sys_path_tokens
        or any(not _is_sha256(value) for value in final_sys_path_tokens)
    ):
        _reject("runtime_binding_authority_invalid")
    route_hashes = {
        name: contract.canonical_sha256(arguments)
        for name, arguments in RUNTIME_ROUTE_ARGUMENTS.items()
    }
    tokens = list(final_sys_path_tokens)
    body = {
        "schema_version": RUNTIME_BINDING_AUTHORITY_SCHEMA_VERSION,
        "repository_runtime_manifest_sha256": _sha256_value(
            repository_runtime_manifest_sha256,
            "runtime_binding_authority_invalid",
        ),
        "execution_dependency_manifest_sha256": _sha256_value(
            execution_dependency_manifest_sha256,
            "runtime_binding_authority_invalid",
        ),
        "checkpoint_loaded_code_manifest_sha256s": {
            name: checkpoints[name] for name in RUNTIME_CHECKPOINT_NAMES
        },
        "route_argv_sha256s": route_hashes,
        "environment_allowlist": list(RUNTIME_ENVIRONMENT_ALLOWLIST),
        "process_environment_sha256": _sha256_value(
            process_environment_sha256, "runtime_binding_authority_invalid"
        ),
        "final_sys_path_tokens": tokens,
        "final_sys_path_sha256": contract.canonical_sha256(tokens),
    }
    return {
        **body,
        "runtime_binding_authority_sha256": contract.canonical_sha256(body),
    }


def validate_runtime_binding_authority(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "repository_runtime_manifest_sha256",
            "execution_dependency_manifest_sha256",
            "checkpoint_loaded_code_manifest_sha256s",
            "route_argv_sha256s",
            "environment_allowlist",
            "process_environment_sha256",
            "final_sys_path_tokens",
            "final_sys_path_sha256",
            "runtime_binding_authority_sha256",
        }
    )
    item = _strict_mapping(value, fields, code="runtime_binding_authority_invalid")
    _validate_self_hash(
        item,
        "runtime_binding_authority_sha256",
        code="runtime_binding_authority_invalid",
    )
    rebuilt = build_runtime_binding_authority(
        repository_runtime_manifest_sha256=item[
            "repository_runtime_manifest_sha256"
        ],
        execution_dependency_manifest_sha256=item[
            "execution_dependency_manifest_sha256"
        ],
        checkpoint_loaded_code_manifest_sha256s=item[
            "checkpoint_loaded_code_manifest_sha256s"
        ],
        process_environment_sha256=item["process_environment_sha256"],
        final_sys_path_tokens=item["final_sys_path_tokens"],
    )
    if item != rebuilt:
        _reject("runtime_binding_authority_invalid")
    return item


def build_interpreter_identity(
    *,
    executable_sha256: str,
    python_dll_sha256: str,
    python_version: str,
    cache_tag: str,
    launcher_profile_sha256: str,
    flags_sha256: str,
    process_environment_sha256: str,
    final_sys_path_sha256: str,
) -> str:
    body = {
        "schema_version": INTERPRETER_IDENTITY_SCHEMA_VERSION,
        "executable_sha256": _sha256_value(executable_sha256, "runtime_interpreter_invalid"),
        "python_dll_sha256": _sha256_value(python_dll_sha256, "runtime_interpreter_invalid"),
        "python_version": python_version,
        "cache_tag": cache_tag,
        "launcher_profile_sha256": _sha256_value(
            launcher_profile_sha256, "runtime_interpreter_invalid"
        ),
        "flags_sha256": _sha256_value(flags_sha256, "runtime_interpreter_invalid"),
        "process_environment_sha256": _sha256_value(
            process_environment_sha256, "runtime_interpreter_invalid"
        ),
        "final_sys_path_sha256": _sha256_value(
            final_sys_path_sha256, "runtime_interpreter_invalid"
        ),
    }
    if type(python_version) is not str or not python_version or type(cache_tag) is not str:
        _reject("runtime_interpreter_invalid")
    return contract.canonical_sha256(body)


def build_runtime_binding_receipt(
    *,
    checkpoint_name: str,
    invocation_kind: str,
    head: str,
    tree: str,
    clean_state_sha256: str,
    authority: Mapping[str, Any],
    interpreter_identity_sha256: str,
    previous_runtime_binding_receipt_sha256: str | None,
) -> dict[str, Any]:
    runtime = validate_runtime_binding_authority(authority)
    if (
        checkpoint_name not in RUNTIME_CHECKPOINT_NAMES
        or invocation_kind not in RUNTIME_ROUTE_ARGUMENTS
        or not _is_sha1(head)
        or not _is_sha1(tree)
        or not _is_sha256(clean_state_sha256)
        or not _is_sha256(interpreter_identity_sha256)
        or (
            previous_runtime_binding_receipt_sha256 is not None
            and not _is_sha256(previous_runtime_binding_receipt_sha256)
        )
    ):
        _reject("runtime_binding_receipt_invalid")
    body = {
        "schema_version": RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION,
        "checkpoint_name": checkpoint_name,
        "invocation_kind": invocation_kind,
        "head": head,
        "tree": tree,
        "clean_state_sha256": clean_state_sha256,
        "repository_runtime_manifest_sha256": runtime[
            "repository_runtime_manifest_sha256"
        ],
        "execution_dependency_manifest_sha256": runtime[
            "execution_dependency_manifest_sha256"
        ],
        "loaded_code_manifest_sha256": runtime[
            "checkpoint_loaded_code_manifest_sha256s"
        ][checkpoint_name],
        "runtime_binding_authority_sha256": runtime[
            "runtime_binding_authority_sha256"
        ],
        "route_argv_sha256": runtime["route_argv_sha256s"][invocation_kind],
        "process_environment_sha256": runtime["process_environment_sha256"],
        "interpreter_identity_sha256": interpreter_identity_sha256,
        "previous_runtime_binding_receipt_sha256": (
            previous_runtime_binding_receipt_sha256
        ),
    }
    return {
        **body,
        "runtime_binding_receipt_sha256": contract.canonical_sha256(body),
    }


def _validate_source_inventory(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(contract.IMPLEMENTATION_ALLOWED_PATHS):
        _reject("preflight_source_inventory_invalid")
    rows: list[dict[str, Any]] = []
    for expected_path, candidate in zip(contract.IMPLEMENTATION_ALLOWED_PATHS, value):
        row = _strict_mapping(
            candidate,
            _SOURCE_INVENTORY_ROW_FIELDS,
            code="preflight_source_inventory_invalid",
        )
        if (
            row["path"] != expected_path
            or not _is_sha1(row["git_blob_sha1"])
            or not _is_sha256(row["literal_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
        ):
            _reject("preflight_source_inventory_invalid")
        rows.append(row)
    return rows


def validate_local_production_closure_manifest(value: Any) -> dict[str, Any]:
    """Validate the exact successor-base local-module closure."""

    item = _strict_mapping(
        value,
        frozenset(
            {
                "base_commit",
                "base_tree",
                "closure_path_count",
                "derivation",
                "explicit_dynamic_local_paths",
                "ordered_roots",
                "paths",
                "schema",
            }
        ),
        code="preflight_local_closure_shape_invalid",
    )
    if (
        item["base_commit"] != contract.BASE_COMMIT
        or item["base_tree"] != contract.BASE_TREE
        or item["closure_path_count"] != len(contract.LOCAL_PRODUCTION_CLOSURE_PATHS)
        or item["derivation"] != contract.LOCAL_PRODUCTION_CLOSURE_DERIVATION
        or item["schema"] != contract.LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION
        or item["ordered_roots"]
        != list(contract.LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS)
        or item["explicit_dynamic_local_paths"]
        != list(contract.LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS)
        or type(item["paths"]) is not list
        or len(item["paths"]) != len(contract.LOCAL_PRODUCTION_CLOSURE_PATHS)
    ):
        _reject("preflight_local_closure_invalid")
    rows: list[dict[str, str]] = []
    for expected_path, candidate in zip(
        contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
        item["paths"],
        strict=True,
    ):
        row = _strict_mapping(
            candidate,
            frozenset({"git_blob_sha1", "literal_sha256", "path"}),
            code="preflight_local_closure_invalid",
        )
        if (
            row["path"] != expected_path
            or not _is_sha1(row["git_blob_sha1"])
            or not _is_sha256(row["literal_sha256"])
        ):
            _reject("preflight_local_closure_invalid")
        rows.append(row)
    normalized = {**item, "paths": rows}
    manifest_bytes = contract.canonical_json_bytes(normalized)
    path_bytes = contract.canonical_json_bytes(
        [row["path"] for row in rows]
    )
    if (
        len(manifest_bytes) != contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES
        or hashlib.sha256(manifest_bytes).hexdigest()
        != contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
        or len(path_bytes) != contract.LOCAL_PRODUCTION_CLOSURE_PATHS_BYTES
        or hashlib.sha256(path_bytes).hexdigest()
        != contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
    ):
        _reject("preflight_local_closure_hash_invalid")
    return normalized


def validate_repository_snapshot(value: Any) -> dict[str, Any]:
    """Validate exact pushed implementation ancestry and source inventory."""

    item = _strict_mapping(
        value,
        _REPOSITORY_SNAPSHOT_FIELDS,
        code="preflight_repository_shape_invalid",
    )
    try:
        ancestry = contract.validate_implementation_ancestry(item["ancestry"])
    except contract.ContractViolation:
        _reject("preflight_implementation_ancestry_invalid")
    rows = _validate_source_inventory(item["source_inventory"])
    production_rows = rows[: len(contract.IMPLEMENTATION_PRODUCTION_PATHS)]
    test_rows = rows[len(contract.IMPLEMENTATION_PRODUCTION_PATHS) :]
    expected_all = contract.canonical_sha256(rows)
    expected_production = contract.canonical_sha256(production_rows)
    expected_tests = contract.canonical_sha256(test_rows)
    try:
        local_import_paths = contract.validate_local_import_paths(
            item["local_import_paths"]
        )
    except contract.ContractViolation:
        _reject("preflight_local_import_upper_bound_invalid")
    required_local_imports = {
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
    }
    if (
        item["source_inventory_sha256"] != expected_all
        or item["production_source_inventory_sha256"] != expected_production
        or item["test_source_inventory_sha256"] != expected_tests
        or not _is_sha256(item["predecessor_inventory_sha256"])
        or item["unchanged_predecessor_inventory_sha256"]
        != item["predecessor_inventory_sha256"]
        or item["local_production_closure_manifest_sha256"]
        != contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
        or item["local_production_closure_paths_sha256"]
        != contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
        or not required_local_imports.issubset(local_import_paths)
        or tuple(
            sorted(local_import_paths, key=lambda value: value.encode("utf-8"))
        )
        != local_import_paths
        or item["local_import_paths_sha256"]
        != contract.canonical_sha256(list(local_import_paths))
        or item["package_initializer_predecessor_equal"] is not True
        or item["preregistration_authenticated"] is not True
    ):
        _reject("preflight_repository_identity_invalid")
    return {
        **item,
        "ancestry": ancestry,
        "source_inventory": rows,
        "local_import_paths": list(local_import_paths),
    }


def validate_bridge_manifest(value: Any) -> dict[str, Any]:
    """Accept only the aggregate, exact-development bridge receipt."""

    item = _strict_mapping(
        value,
        _BRIDGE_MANIFEST_FIELDS,
        code="preflight_bridge_manifest_shape_invalid",
    )
    body = dict(item)
    observed_hash = body.pop("bridge_sha256", None)
    hash_fields = {
        "source_authority_pins_sha256",
        "source_inventory_sha256",
        "stage_source_seal_sha256",
        "checkpoint_sha256",
        "compact_replay_sha256",
        "role_manifests_sha256",
        "role_plan_sha256",
        "science_projection_sha256",
        "legacy_projection_sha256",
        "compatibility_manifest_sha256",
        "universe_sha256",
        "content_manifest_sha256",
        "calendar_sessions_sha256",
        "universe_event_proofs_sha256",
        "documents_sha256",
        "records_sha256",
        "source_order_sha256",
        "event_order_sha256",
        "prior_links_sha256",
        "primary_documents_sha256",
    }
    if any(not _is_sha256(item[field]) for field in hash_fields):
        _reject("preflight_bridge_manifest_hash_invalid")
    if (
        not _is_sha256(observed_hash)
        or observed_hash != contract.canonical_sha256(body)
        or item["schema_version"] != BRIDGE_MANIFEST_SCHEMA_VERSION
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["document_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["filename_present_count"]
        != contract.DEVELOPMENT_FILENAME_PRESENT_COUNT
        or item["filename_missing_count"]
        != contract.DEVELOPMENT_FILENAME_MISSING_COUNT
        or item["source_authority_pins_sha256"]
        != contract.canonical_sha256(contract.build_source_authority_pins())
        or item["source_authority_base_commit"] != contract.V38_SOURCE_COMMIT
        or item["source_authority_base_tree"] != contract.V38_SOURCE_TREE
        or item["source_inventory_sha256"] != contract.V38_INVENTORY_SHA256
        or item["stage_source_seal_sha256"] != contract.V38_STAGE_SOURCE_SEAL_SHA256
        or item["checkpoint_sha256"] != contract.V38_LOGICAL_CHECKPOINT_SHA256
        or item["compact_replay_sha256"] != contract.V38_COMPACT_REPLAY_SHA256
        or item["role_manifests_sha256"]
        != contract.V38_ROLE_MANIFEST_INVENTORY_SHA256
        or item["role_plan_sha256"] != contract.V38_ROLE_PLAN_SHA256
        or item["science_projection_sha256"] != contract.SCIENCE_PROJECTION_SHA256
        or item["calendar_sessions_sha256"]
        != contract.CALENDAR_SESSIONS_SHA256
        or item["set_parity"] is not True
        or item["source_sequence_parity"] is not True
        or item["legacy_projection_parity"] is not True
        or item["typed_identity_parity"] is not True
        or item["nullable_filename_parity"] is not True
        or item["no_fabricated_primary_url"] is not True
        or item["prior_links_are_internal_only"] is not True
        or item["first_10k_and_10q_have_no_prior"] is not True
        or item["peak_live_complete_submission_blob_count"] != 1
        or item["sec_request_count"] != 0
        or item["confirmation_or_final_opened"] is not False
        or item["contains_private_rows"] is not False
        or item["contains_accessions_urls_filenames_or_bodies"] is not False
    ):
        _reject("preflight_bridge_manifest_invalid")
    return item


def validate_request_commitments(value: Any) -> dict[str, Any]:
    """Validate the exact aggregate-only 75-request/pilot commitment."""

    item = _strict_mapping(
        value,
        _REQUEST_COMMITMENT_FIELDS,
        code="preflight_request_commitments_shape_invalid",
    )
    hash_fields = {
        "documents_sha256",
        "records_sha256",
        "events_sha256",
        "compatibility_manifest_sha256",
        "universe_sha256",
        "content_manifest_sha256",
        "calendar_sessions_sha256",
        "universe_event_proofs_sha256",
        "preprocessed_events_sha256",
        "canonical_requests_sha256",
        "model_slice_sha256",
        "canonical_request_index_sha256",
        "model_plan_sha256",
        "pilot_order_sha256",
        "remaining_order_sha256",
        "source_order_sha256",
        "prior_links_sha256",
    }
    if any(not _is_sha256(item[field]) for field in hash_fields):
        _reject("preflight_request_commitments_hash_invalid")
    if (
        item["schema_version"] != REQUEST_COMMITMENTS_SCHEMA_VERSION
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["document_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["record_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["event_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["request_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or item["remaining_count"] != contract.DEVELOPMENT_REMAINING_COUNT
        or item["filename_present_count"]
        != contract.DEVELOPMENT_FILENAME_PRESENT_COUNT
        or item["filename_missing_count"]
        != contract.DEVELOPMENT_FILENAME_MISSING_COUNT
        or item["calendar_sessions_sha256"]
        != contract.CALENDAR_SESSIONS_SHA256
        or type(item["minimum_request_byte_count"]) is not int
        or type(item["maximum_request_byte_count"]) is not int
        or not (
            0
            < item["minimum_request_byte_count"]
            <= item["maximum_request_byte_count"]
            <= contract.MODEL_INPUT_MAX_BYTES
        )
        or item["set_parity"] is not True
        or item["sequence_parity"] is not True
        or item["prior_link_parity"] is not True
        or item["confirmation_or_final_opened"] is not False
        or item["contains_private_rows"] is not False
    ):
        _reject("preflight_request_commitments_invalid")
    return item


def _validate_bridge_commitment_parity(
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
) -> None:
    if (
        commitments["document_count"] != bridge["document_count"]
        or commitments["filename_present_count"]
        != bridge["filename_present_count"]
        or commitments["filename_missing_count"]
        != bridge["filename_missing_count"]
        or
        commitments["documents_sha256"] != bridge["documents_sha256"]
        or commitments["records_sha256"] != bridge["records_sha256"]
        or commitments["events_sha256"] != bridge["event_order_sha256"]
        or commitments["compatibility_manifest_sha256"]
        != bridge["compatibility_manifest_sha256"]
        or commitments["universe_sha256"] != bridge["universe_sha256"]
        or commitments["content_manifest_sha256"]
        != bridge["content_manifest_sha256"]
        or commitments["calendar_sessions_sha256"]
        != bridge["calendar_sessions_sha256"]
        or commitments["universe_event_proofs_sha256"]
        != bridge["universe_event_proofs_sha256"]
        or commitments["source_order_sha256"] != bridge["source_order_sha256"]
        or commitments["prior_links_sha256"] != bridge["prior_links_sha256"]
    ):
        _reject("preflight_bridge_request_parity_invalid")


def validate_qualification_report(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _QUALIFICATION_REPORT_FIELDS,
        code="preflight_qualification_shape_invalid",
    )
    _validate_self_hash(
        item,
        "qualification_sha256",
        code="preflight_qualification_hash_invalid",
    )
    if (
        item["schema_version"] != QUALIFICATION_REPORT_SCHEMA_VERSION
        or item["suite_identity"] != contract.QUALIFICATION_SUITE_ID
        or item["status"] != "passed"
        or item["phase_count"] != len(contract.QUALIFICATION_PHASES)
        or item["deadline_seconds"] != contract.QUALIFICATION_TIMEOUT_SECONDS
        or item["durability_mode"] != contract.QUALIFICATION_DURABILITY_MODE
        or item["command_profile_sha256"] != QUALIFICATION_COMMAND_PROFILE_SHA256
        or not _is_sha256(item["private_aggregate_sha256"])
        or type(item["phases"]) is not list
        or len(item["phases"]) != len(contract.QUALIFICATION_PHASES)
    ):
        _reject("preflight_qualification_failed")
    phases: list[dict[str, Any]] = []
    for expected_phase, candidate in zip(
        contract.QUALIFICATION_PHASES,
        item["phases"],
        strict=True,
    ):
        phase = _strict_mapping(
            candidate,
            _QUALIFICATION_PUBLIC_PHASE_FIELDS,
            code="preflight_qualification_shape_invalid",
        )
        numeric = (
            "node_count",
            "collection_duration_ns",
            "execution_duration_ns",
            "passed_count",
            "failed_count",
            "error_count",
            "skipped_count",
            "xfailed_count",
            "xpassed_count",
        )
        hash_fields = (
            "node_list_sha256",
            "collection_result_sha256",
            "execution_result_sha256",
            "collection_log_sha256",
            "execution_log_sha256",
            "collection_xml_sha256",
            "execution_xml_sha256",
            "collection_manifest_sha256",
        )
        if (
            phase["phase"] != expected_phase
            or any(
                type(phase[field]) is not int or phase[field] < 0
                for field in numeric
            )
            or phase["collection_exit_code"] != 0
            or phase["execution_exit_code"] != 0
            or any(not _is_sha256(phase[field]) for field in hash_fields)
            or phase["passed_count"] != phase["node_count"]
            or any(
                phase[field] != 0
                for field in (
                    "failed_count",
                    "error_count",
                    "skipped_count",
                    "xfailed_count",
                    "xpassed_count",
                )
            )
        ):
            _reject("preflight_qualification_failed")
        if expected_phase == contract.QUALIFICATION_PHASE_V315:
            if phase["node_count"] < contract.QUALIFICATION_LATEST_MIN_NODE_COUNT:
                _reject("preflight_qualification_failed")
        elif expected_phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
            if (
                phase["node_count"] != contract.QUALIFICATION_SHARED_NODE_COUNT
                or phase["node_list_sha256"]
                != contract.QUALIFICATION_SHARED_NODE_LIST_SHA256
            ):
                _reject("preflight_qualification_failed")
        elif (
            phase["node_count"] != contract.QUALIFICATION_SENTINEL_NODE_COUNT
            or phase["node_list_sha256"]
            != contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
        ):
            _reject("preflight_qualification_failed")
        phases.append(phase)
    return {**item, "phases": phases}


def _validate_effect_snapshot(value: Any) -> dict[str, int]:
    try:
        return contract.validate_effect_counts(value, route="zero_effect_preflight")
    except contract.ContractViolation:
        _reject("preflight_nonzero_effect")


def _privacy_token_bytes(
    root: Path,
    provider: Callable[[Path], Sequence[bytes | str]],
) -> tuple[bytes, ...]:
    try:
        raw = provider(root)
    except Exception:
        _reject("preflight_privacy_tokens_unavailable")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        _reject("preflight_privacy_tokens_invalid")
    tokens: list[bytes] = []
    for item in raw:
        # bytes.lower() is reliably case-insensitive only for ASCII.  Refuse
        # an unscannable secret instead of silently weakening the privacy gate.
        if isinstance(item, str):
            try:
                token = item.encode("ascii")
            except UnicodeError:
                _reject("preflight_privacy_tokens_invalid")
        elif type(item) is bytes:
            token = item
        else:
            _reject("preflight_privacy_tokens_invalid")
        if not token or not token.isascii():
            _reject("preflight_privacy_tokens_invalid")
        tokens.append(token)
    if not tokens:
        _reject("preflight_privacy_tokens_invalid")
    return tuple(tokens)


def _scan_private_bytes(payload: bytes, forbidden: Sequence[bytes]) -> dict[str, bool]:
    if _payload_contains_forbidden_token(payload, forbidden):
        _reject("preflight_private_privacy_scan_failed")
    return {
        "forbidden_token_echo_absent": True,
        "absolute_private_path_absent": True,
        "readable_contact_absent": True,
    }


def _scan_public_bytes(payload: bytes, forbidden: Sequence[bytes]) -> dict[str, bool]:
    lowered = payload.lower()
    if (
        _payload_contains_forbidden_token(payload, forbidden)
        or _ACCESSION_RE.search(payload) is not None
        or b"http://" in lowered
        or b"https://" in lowered
        or b"canonical_request_body" in lowered
        or b"response_body" in lowered
        or b"normalized_text" in lowered
        or b"selected_filename" in lowered
    ):
        _reject("preflight_public_privacy_scan_failed")
    return {
        "forbidden_token_echo_absent": True,
        "accessions_absent": True,
        "urls_absent": True,
        "filenames_offsets_bodies_absent": True,
        "private_paths_absent": True,
        "pre_release_science_values_absent": True,
    }


def _private_tree_forbidden_tokens(
    root: Path,
    full_tokens: Sequence[bytes],
) -> tuple[bytes, ...]:
    """Allow bound repo/v3.15 paths privately; keep contact and v3.8 forbidden."""

    allowed_paths = {
        root,
        root / Path(contract.PRIVATE_NAMESPACE),
        root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE),
        root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE),
    }
    allowed = {
        str(path.resolve(strict=False))
        .replace("\\", "/")
        .rstrip("/")
        .casefold()
        for path in allowed_paths
    }
    private: list[bytes] = []
    for token in full_tokens:
        if type(token) is not bytes or not token:
            _reject("preflight_privacy_tokens_invalid")
        try:
            decoded = token.decode("ascii", errors="strict")
        except UnicodeError:
            _reject("preflight_privacy_tokens_invalid")
        normalized = decoded.replace("\\", "/").rstrip("/").casefold()
        if normalized not in allowed:
            private.append(token)
    if not private:
        _reject("preflight_privacy_tokens_invalid")
    return tuple(private)


def _contact_v38_private_tokens(root: Path, contact: str) -> tuple[bytes, ...]:
    return _privacy_token_bytes(
        root,
        lambda _path: (
            contact,
            str((root / V38_PRIVATE_ROOT).resolve(strict=False)),
        ),
    )


def _json_strings(value: Any) -> list[str]:
    """Return every parsed JSON key and string value for privacy scanning."""

    strings: list[str] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            strings.append(item)
        elif type(item) is dict:
            for key, child in item.items():
                if type(key) is str:
                    strings.append(key)
                pending.append(child)
        elif type(item) is list:
            pending.extend(item)
    return strings


def _privacy_text_forms(value: str) -> frozenset[str]:
    """Normalize the spellings a private value can take inside JSON.

    In particular, Windows paths may be serialized with doubled backslashes,
    forward slashes, changed case, or ``\\uXXXX`` escapes.  Looking only for
    the original UTF-8 bytes would miss those ordinary JSON representations.
    """

    slash = value.replace("\\", "/")
    backslash = value.replace("/", "\\")
    raw = {value, slash, backslash}
    forms = {item.casefold() for item in raw if item}
    for item in raw:
        if not item:
            continue
        for ensure_ascii in (False, True):
            encoded = json.dumps(item, ensure_ascii=ensure_ascii)[1:-1]
            forms.add(encoded.casefold())
    return frozenset(forms)


def _payload_contains_forbidden_token(
    payload: bytes, forbidden: Sequence[bytes]
) -> bool:
    """Fail-closed token search over raw, JSON-escaped, and parsed strings."""

    if type(payload) is not bytes:
        return True
    byte_forms, text_forms = _forbidden_token_forms(forbidden)
    if not byte_forms:
        return True
    lowered = payload.lower()
    if any(item in lowered for item in byte_forms):
        return True
    try:
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, TypeError):
        return False
    for candidate in _json_strings(parsed):
        candidate_forms = _privacy_text_forms(candidate)
        if any(
            token in candidate_form
            for candidate_form in candidate_forms
            for token in text_forms
        ):
            return True
    return False


def _forbidden_token_forms(
    forbidden: Sequence[bytes],
) -> tuple[frozenset[bytes], frozenset[str]]:
    byte_forms: set[bytes] = set()
    text_forms: set[str] = set()
    for token in forbidden:
        if type(token) is not bytes or not token:
            return frozenset(), frozenset()
        byte_forms.add(token.lower())
        try:
            decoded = token.decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        for form in _privacy_text_forms(decoded):
            text_forms.add(form)
            byte_forms.add(form.encode("utf-8"))
    return frozenset(byte_forms), frozenset(text_forms)


def _stream_file_sha256_and_privacy(
    path: Path,
    *,
    byte_forms: frozenset[bytes],
    code: str,
) -> tuple[str, int]:
    """Hash a private file while scanning tokens across chunk boundaries."""

    maximum = max((len(item) for item in byte_forms), default=0)
    overlap = b""
    digest = hashlib.sha256()
    byte_count = 0
    before = _optional_lstat(path, code=code)
    if before is None or not _ordinary_regular_stat(before):
        _reject(code)
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(_TREE_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
                if byte_forms:
                    searchable = (overlap + chunk).lower()
                    if any(item in searchable for item in byte_forms):
                        _reject(code)
                    overlap = (
                        searchable[-(maximum - 1) :] if maximum > 1 else b""
                    )
            opened_after = os.fstat(handle.fileno())
    except V315PreflightError:
        raise
    except OSError:
        _reject(code)
    after = _optional_lstat(path, code=code)
    if (
        after is None
        or not _ordinary_regular_stat(opened_before)
        or not _ordinary_regular_stat(opened_after)
        or not _ordinary_regular_stat(after)
        or any(
            details.st_size != byte_count
            for details in (before, opened_before, opened_after, after)
        )
        or _regular_file_identity(before)
        != _regular_file_identity(opened_before)
        or _regular_file_identity(opened_before)
        != _regular_file_identity(opened_after)
        or _descriptor_file_identity(opened_before)
        != _descriptor_file_identity(opened_after)
        or _regular_file_identity(opened_after)
        != _regular_file_identity(after)
    ):
        _reject(code)
    return digest.hexdigest(), byte_count


def _snapshot_serialized_tree(
    directory: Path,
    *,
    repo_root: Path,
    forbidden: Sequence[bytes] | None,
    code: str,
) -> dict[str, Any]:
    """Hash/stat a tree without following links or retaining file bodies."""

    try:
        resolved = directory.resolve(strict=True)
        resolved.relative_to(repo_root)
        root_details = directory.lstat()
    except (OSError, ValueError):
        _reject(code)
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or getattr(root_details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        _reject(code)
    byte_forms: frozenset[bytes] = frozenset()
    if forbidden is not None:
        byte_forms, _text_forms = _forbidden_token_forms(forbidden)
        if not byte_forms:
            _reject(code)
    rows: list[dict[str, Any]] = []
    pending = [directory]
    while pending:
        parent = pending.pop()
        try:
            entries = sorted(os.scandir(parent), key=lambda item: item.name)
        except OSError:
            _reject(code)
        for entry in entries:
            path = Path(entry.path)
            try:
                details = entry.stat(follow_symlinks=False)
                relative = path.relative_to(directory).as_posix()
            except (OSError, ValueError):
                _reject(code)
            if (
                not relative
                or entry.is_symlink()
                or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                _reject(code)
            if forbidden is not None and _payload_contains_forbidden_token(
                relative.encode("utf-8", errors="strict"), forbidden
            ):
                _reject(code)
            common = {
                "path": relative,
                "mode": int(details.st_mode),
                "mtime_ns": int(details.st_mtime_ns),
            }
            if stat.S_ISDIR(details.st_mode):
                rows.append({**common, "kind": "directory"})
                pending.append(path)
            elif stat.S_ISREG(details.st_mode):
                digest, observed_bytes = _stream_file_sha256_and_privacy(
                    path, byte_forms=byte_forms, code=code
                )
                if observed_bytes != details.st_size:
                    _reject(code)
                rows.append(
                    {
                        **common,
                        "kind": "file",
                        "byte_count": observed_bytes,
                        "literal_sha256": digest,
                    }
                )
            else:
                _reject(code)
    rows.sort(key=lambda item: item["path"])
    return {
        "entry_count": len(rows),
        "inventory_sha256": contract.canonical_sha256(rows),
        "rows": rows,
    }


def _scan_qualification_private_tree(
    root: Path,
    *,
    forbidden: Sequence[bytes],
    code: str,
) -> dict[str, Any]:
    """Scan and hash every sealed qualification receipt."""

    return _snapshot_serialized_tree(
        root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE) / "qualification",
        repo_root=root,
        forbidden=forbidden,
        code=code,
    )


def _canonical_repo_root(value: Path) -> Path:
    if not isinstance(value, Path):
        _reject("preflight_repo_root_invalid")
    try:
        root = value.resolve(strict=True)
    except OSError:
        _reject("preflight_repo_root_invalid")
    if not root.is_dir() or not (root / ".git").exists():
        _reject("preflight_repo_root_invalid")
    return root


def _assert_regular_new_parent(path: Path, root: Path) -> None:
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
        details = resolved_parent.lstat()
    except (OSError, ValueError):
        _reject("preflight_output_parent_invalid")
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        _reject("preflight_output_parent_invalid")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = [
                _wintypes.LPCWSTR,
                _wintypes.DWORD,
                _wintypes.DWORD,
                _wintypes.LPVOID,
                _wintypes.DWORD,
                _wintypes.DWORD,
                _wintypes.HANDLE,
            ]
            kernel32.CreateFileW.restype = _wintypes.HANDLE
            kernel32.FlushFileBuffers.argtypes = [_wintypes.HANDLE]
            kernel32.FlushFileBuffers.restype = _wintypes.BOOL
            kernel32.CloseHandle.argtypes = [_wintypes.HANDLE]
            kernel32.CloseHandle.restype = _wintypes.BOOL
            handle = kernel32.CreateFileW(
                str(path.resolve(strict=True)),
                0x40000000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000,
                None,
            )
            invalid = ctypes.c_void_p(-1).value
            if handle in (None, invalid):
                raise OSError(ctypes.get_last_error(), "CreateFileW")
            try:
                if not kernel32.FlushFileBuffers(handle):
                    raise OSError(ctypes.get_last_error(), "FlushFileBuffers")
            finally:
                if not kernel32.CloseHandle(handle):
                    raise OSError(ctypes.get_last_error(), "CloseHandle")
            return
        except OSError:
            _reject("preflight_directory_fsync_failed")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optional_lstat(path: Path, *, code: str) -> os.stat_result | None:
    """Return an entry's own metadata without following a broken link."""

    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        _reject(code)


def _regular_file_identity(details: os.stat_result) -> tuple[int, ...]:
    """Return every stable file identity field used by descriptor reads."""

    return (
        details.st_dev,
        details.st_ino,
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        getattr(details, "st_file_attributes", 0),
    )


def _descriptor_file_identity(details: os.stat_result) -> tuple[int, ...]:
    """Add descriptor-consistent change time to the visible identity."""

    return (*_regular_file_identity(details), details.st_mode, details.st_ctime_ns)


def _ordinary_regular_stat(
    details: os.stat_result, *, allow_hardlinks: bool = False
) -> bool:
    return (
        stat.S_ISREG(details.st_mode)
        and not stat.S_ISLNK(details.st_mode)
        and not getattr(details, "st_file_attributes", 0)
        & _REPARSE_ATTRIBUTE
        and (
            details.st_nlink in {0, 1}
            or allow_hardlinks
            and details.st_nlink > 1
        )
        and details.st_size >= 0
    )


def _read_regular_file(
    path: Path,
    *,
    maximum: int | None,
    code: str,
    allow_hardlinks: bool = False,
) -> bytes:
    """Read one ordinary, unlinked file and reject replacement races."""

    before = _optional_lstat(path, code=code)
    if before is None:
        _reject(code)
    if not _ordinary_regular_stat(before, allow_hardlinks=allow_hardlinks) or (
        maximum is not None and before.st_size > maximum
    ):
        _reject(code)
    try:
        chunks: list[bytes] = []
        total = 0
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(_TREE_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if maximum is not None and total > maximum:
                    _reject(code)
                chunks.append(chunk)
            opened_after = os.fstat(handle.fileno())
        payload = b"".join(chunks)
    except V315PreflightError:
        raise
    except OSError:
        _reject(code)
    after = _optional_lstat(path, code=code)
    if (
        after is None
        or not _ordinary_regular_stat(
            opened_before, allow_hardlinks=allow_hardlinks
        )
        or not _ordinary_regular_stat(
            opened_after, allow_hardlinks=allow_hardlinks
        )
        or not _ordinary_regular_stat(after, allow_hardlinks=allow_hardlinks)
        or before.st_size != len(payload)
        or opened_before.st_size != len(payload)
        or opened_after.st_size != len(payload)
        or after.st_size != len(payload)
        or _regular_file_identity(before)
        != _regular_file_identity(opened_before)
        or _regular_file_identity(opened_before)
        != _regular_file_identity(opened_after)
        or _descriptor_file_identity(opened_before)
        != _descriptor_file_identity(opened_after)
        or _regular_file_identity(opened_after)
        != _regular_file_identity(after)
    ):
        _reject(code)
    return payload


def _assert_exact_regular_file(path: Path, payload: bytes, *, code: str) -> None:
    if _read_regular_file(path, maximum=len(payload), code=code) != payload:
        _reject(code)


def _write_atomic_new_file(path: Path, payload: bytes, *, code: str) -> None:
    """Durably stage bytes and atomically promote them to a new public path."""

    pending = path.with_name(f"{path.name}.v315-pending")
    final_state = _optional_lstat(path, code=code)
    pending_state = _optional_lstat(pending, code=code)
    if final_state is not None:
        if pending_state is not None:
            _reject(code)
        _assert_exact_regular_file(path, payload, code=code)
        return

    if pending_state is None:
        try:
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            # A simultaneous recovery may have staged the same immutable bytes.
            pass
        except OSError:
            _reject(code)
    _assert_exact_regular_file(pending, payload, code=code)
    _fsync_directory(path.parent)
    if _optional_lstat(path, code=code) is not None:
        _reject(code)
    try:
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except OSError:
        _reject(code)
    _assert_exact_regular_file(path, payload, code=code)
    if _optional_lstat(pending, code=code) is not None:
        _reject(code)


def _write_new_file(path: Path, payload: bytes, *, code: str) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
            or _read_regular_file(path, maximum=len(payload), code=code)
            != payload
        ):
            _reject(code)
    except FileExistsError:
        _reject("preflight_already_consumed")
    except V315PreflightError:
        raise
    except OSError:
        _reject(code)


def _preflight_intent() -> dict[str, Any]:
    return _self_hash(
        {
            "schema_version": PREFLIGHT_INTENT_SCHEMA_VERSION,
            "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
            "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
            "preregistration_commit": contract.PREREGISTRATION_COMMIT,
            "external_effects_authorized": False,
        },
        "intent_sha256",
    )


def _reserve_once(
    root: Path,
    *,
    on_consumption: Callable[[Path], None] | None = None,
) -> Path:
    public_path = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    if (
        _optional_lstat(public_path, code="preflight_already_consumed") is not None
        or _optional_lstat(private_root, code="preflight_already_consumed") is not None
    ):
        _reject("preflight_already_consumed")
    parent = private_root.parent
    if on_consumption is not None:
        # This callback runs immediately before the first reservation mutation.
        # From this point onward even a partial write/fsync failure is consuming.
        on_consumption(private_root)
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_resolved = parent.resolve(strict=True)
        parent_resolved.relative_to(root)
        private_root.mkdir(exist_ok=False)
        (private_root / PRIVATE_MANIFEST_DIRECTORY).mkdir(exist_ok=False)
    except FileExistsError:
        _reject("preflight_already_consumed")
    except (OSError, ValueError):
        _reject("preflight_private_state_unwritable")
    intent = _preflight_intent()
    _write_new_file(
        private_root / PRIVATE_INTENT_FILENAME,
        _artifact_bytes(intent),
        code="preflight_private_state_unwritable",
    )
    _fsync_directory(parent_resolved)
    return private_root


def _projection_manifest(value: Any) -> dict[str, Any]:
    try:
        candidate = getattr(value, "manifest")
    except Exception:
        _reject("preflight_projection_invalid")
    return validate_bridge_manifest(candidate)


def _build_private_manifest(
    *,
    repository: Mapping[str, Any],
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
    effect_before: Mapping[str, int],
    effect_after: Mapping[str, int],
    tests: Mapping[str, Any],
    runtime_authority: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "implementation_ancestry_passed": True,
        "exact_twelve_file_delta_passed": True,
        "predecessor_blobs_unchanged": True,
        "source_authority_authenticated": True,
        "streaming_projection_parity_passed": True,
        "request_and_pilot_commitments_frozen": True,
        "confirmation_and_final_unreachable": True,
        "zero_external_effects": True,
        "qualification_passed": True,
        "runtime_authority_created": True,
        "privacy_passed": True,
        "eligible_for_public_seal": True,
    }
    unsigned = {
        "schema_version": PRIVATE_MANIFEST_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "passed",
        "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
        "repository": copy.deepcopy(dict(repository)),
        "bridge": copy.deepcopy(dict(bridge)),
        "request_commitments": copy.deepcopy(dict(commitments)),
        "effect_counts_before": copy.deepcopy(dict(effect_before)),
        "effect_counts_after": copy.deepcopy(dict(effect_after)),
        "qualification": copy.deepcopy(dict(tests)),
        "runtime_binding_authority_sha256": runtime_authority[
            "runtime_binding_authority_sha256"
        ],
        "privacy": {
            "aggregate_only_bridge_serialized": True,
            "aggregate_only_request_commitments_serialized": True,
            "readable_contact_stored": False,
            "compact_manifest_absolute_private_path_stored": False,
            "qualification_intent_absolute_paths_stored": True,
        },
        "gates": gates,
    }
    return _self_hash(unsigned, "private_manifest_sha256")


def _build_public_artifact(
    *,
    repository: Mapping[str, Any],
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
    effects: Mapping[str, int],
    tests: Mapping[str, Any],
    runtime_authority: Mapping[str, Any],
    private_manifest_sha256: str,
    private_manifest_literal_sha256: str,
) -> dict[str, Any]:
    ancestry = repository["ancestry"]
    gates = {
        "implementation_ancestry_passed": True,
        "exact_twelve_file_delta_passed": True,
        "predecessor_blobs_unchanged": True,
        "source_authority_authenticated": True,
        "streaming_projection_parity_passed": True,
        "request_and_pilot_commitments_frozen": True,
        "confirmation_and_final_unreachable": True,
        "zero_external_effects": True,
        "qualification_passed": True,
        "runtime_authority_created": True,
        "privacy_passed": True,
    }
    unsigned = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "passed",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preregistration_commit": contract.PREREGISTRATION_COMMIT,
        "preregistration_tree": contract.PREREGISTRATION_TREE,
        "implementation_commit": ancestry["commit"],
        "implementation_tree": ancestry["tree"],
        "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "source_authority": {
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "source_authority_literal_sha256": (
                contract.V38_SOURCE_AUTHORITY_LITERAL_SHA256
            ),
            "logical_checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
            "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
            "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
            "role_manifest_inventory_sha256": (
                contract.V38_ROLE_MANIFEST_INVENTORY_SHA256
            ),
            "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
        },
        "counts": {
            "documents": commitments["document_count"],
            "records": commitments["record_count"],
            "events": commitments["event_count"],
            "canonical_requests": commitments["request_count"],
            "pilots": commitments["pilot_count"],
            "remaining": commitments["remaining_count"],
            "filename_present": commitments["filename_present_count"],
            "filename_missing": commitments["filename_missing_count"],
            "minimum_request_byte_count": commitments[
                "minimum_request_byte_count"
            ],
            "maximum_request_byte_count": commitments[
                "maximum_request_byte_count"
            ],
            "implementation_production_files": len(
                contract.IMPLEMENTATION_PRODUCTION_PATHS
            ),
            "implementation_test_files": len(contract.IMPLEMENTATION_TEST_PATHS),
            "local_production_closure_files": len(
                contract.LOCAL_PRODUCTION_CLOSURE_PATHS
            ),
            "local_import_module_count": len(repository["local_import_paths"]),
            "experiment_family_sec_requests": (
                contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
            ),
        },
        "aggregates": {
            "bridge_sha256": bridge["bridge_sha256"],
            "legacy_projection_sha256": bridge["legacy_projection_sha256"],
            "compatibility_manifest_sha256": commitments[
                "compatibility_manifest_sha256"
            ],
            "universe_sha256": commitments["universe_sha256"],
            "content_manifest_sha256": commitments[
                "content_manifest_sha256"
            ],
            "calendar_sessions_sha256": commitments[
                "calendar_sessions_sha256"
            ],
            "universe_event_proofs_sha256": commitments[
                "universe_event_proofs_sha256"
            ],
            "documents_sha256": commitments["documents_sha256"],
            "records_sha256": commitments["records_sha256"],
            "events_sha256": commitments["events_sha256"],
            "preprocessed_events_sha256": commitments[
                "preprocessed_events_sha256"
            ],
            "canonical_requests_sha256": commitments[
                "canonical_requests_sha256"
            ],
            "model_slice_sha256": commitments["model_slice_sha256"],
            "canonical_request_index_sha256": commitments[
                "canonical_request_index_sha256"
            ],
            "model_plan_sha256": commitments["model_plan_sha256"],
            "pilot_order_sha256": commitments["pilot_order_sha256"],
            "remaining_order_sha256": commitments["remaining_order_sha256"],
            "source_order_sha256": commitments["source_order_sha256"],
            "prior_links_sha256": commitments["prior_links_sha256"],
            "production_source_inventory_sha256": repository[
                "production_source_inventory_sha256"
            ],
            "test_source_inventory_sha256": repository[
                "test_source_inventory_sha256"
            ],
            "all_source_inventory_sha256": repository[
                "source_inventory_sha256"
            ],
            "predecessor_inventory_sha256": repository[
                "predecessor_inventory_sha256"
            ],
            "local_production_closure_manifest_sha256": repository[
                "local_production_closure_manifest_sha256"
            ],
            "local_production_closure_order_sha256": repository[
                "local_production_closure_paths_sha256"
            ],
            "local_import_set_sha256": repository[
                "local_import_paths_sha256"
            ],
        },
        "effect_counts": copy.deepcopy(dict(effects)),
        "qualification": copy.deepcopy(dict(tests)),
        "runtime_binding_authority_sha256": runtime_authority[
            "runtime_binding_authority_sha256"
        ],
        "privacy": {
            "readable_contact_published": False,
            "accessions_urls_filenames_offsets_bodies_published": False,
            "private_paths_published": False,
            "canonical_requests_published": False,
            "pre_release_science_values_published": False,
            "redacted_errors_only": True,
        },
        "gates": gates,
        "private_manifest_sha256": private_manifest_sha256,
        "private_manifest_literal_sha256": private_manifest_literal_sha256,
        "eligible_for_public_seal": True,
        "development_authorized": False,
    }
    return _self_hash(unsigned, "public_artifact_sha256")


def validate_private_preflight_manifest(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PRIVATE_MANIFEST_FIELDS,
        code="preflight_private_manifest_shape_invalid",
    )
    _validate_self_hash(item, "private_manifest_sha256", code="preflight_private_manifest_hash_invalid")
    gates = _strict_mapping(
        item["gates"], _PRIVATE_GATES, code="preflight_private_manifest_invalid"
    )
    privacy = _strict_mapping(
        item["privacy"],
        _PRIVATE_PRIVACY_FIELDS,
        code="preflight_private_manifest_invalid",
    )
    if (
        item["schema_version"] != PRIVATE_MANIFEST_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "passed"
        or item["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or not _is_sha256(item["runtime_binding_authority_sha256"])
        or any(child is not True for child in gates.values())
        or privacy
        != {
            "aggregate_only_bridge_serialized": True,
            "aggregate_only_request_commitments_serialized": True,
            "readable_contact_stored": False,
            "compact_manifest_absolute_private_path_stored": False,
            "qualification_intent_absolute_paths_stored": True,
        }
    ):
        _reject("preflight_private_manifest_invalid")
    validate_repository_snapshot(item["repository"])
    bridge = validate_bridge_manifest(item["bridge"])
    commitments = validate_request_commitments(item["request_commitments"])
    _validate_bridge_commitment_parity(bridge, commitments)
    _validate_effect_snapshot(item["effect_counts_before"])
    _validate_effect_snapshot(item["effect_counts_after"])
    validate_qualification_report(item["qualification"])
    return item


def validate_public_preflight_artifact(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUBLIC_ARTIFACT_FIELDS,
        code="preflight_public_artifact_shape_invalid",
    )
    _validate_self_hash(item, "public_artifact_sha256", code="preflight_public_artifact_hash_invalid")
    source_authority = _strict_mapping(
        item["source_authority"],
        frozenset(
            {
                "terminal_internal_sha256",
                "source_authority_literal_sha256",
                "logical_checkpoint_sha256",
                "stage_source_seal_sha256",
                "compact_replay_sha256",
                "role_manifest_inventory_sha256",
                "role_plan_sha256",
                "inventory_sha256",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    counts = _strict_mapping(
        item["counts"],
        frozenset(
            {
                "documents",
                "records",
                "events",
                "canonical_requests",
                "pilots",
                "remaining",
                "filename_present",
                "filename_missing",
                "minimum_request_byte_count",
                "maximum_request_byte_count",
                "implementation_production_files",
                "implementation_test_files",
                "local_production_closure_files",
                "local_import_module_count",
                "experiment_family_sec_requests",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    aggregates = _strict_mapping(
        item["aggregates"],
        frozenset(
            {
                "bridge_sha256",
                "legacy_projection_sha256",
                "compatibility_manifest_sha256",
                "universe_sha256",
                "content_manifest_sha256",
                "calendar_sessions_sha256",
                "universe_event_proofs_sha256",
                "documents_sha256",
                "records_sha256",
                "events_sha256",
                "preprocessed_events_sha256",
                "canonical_requests_sha256",
                "model_slice_sha256",
                "canonical_request_index_sha256",
                "model_plan_sha256",
                "pilot_order_sha256",
                "remaining_order_sha256",
                "source_order_sha256",
                "prior_links_sha256",
                "production_source_inventory_sha256",
                "test_source_inventory_sha256",
                "all_source_inventory_sha256",
                "predecessor_inventory_sha256",
                "local_production_closure_manifest_sha256",
                "local_production_closure_order_sha256",
                "local_import_set_sha256",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    privacy = _strict_mapping(
        item["privacy"],
        _PUBLIC_PRIVACY_FIELDS,
        code="preflight_public_artifact_invalid",
    )
    gates = _strict_mapping(
        item["gates"], _PUBLIC_GATES, code="preflight_public_artifact_invalid"
    )
    expected_source = {
        "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
        "source_authority_literal_sha256": contract.V38_SOURCE_AUTHORITY_LITERAL_SHA256,
        "logical_checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
        "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
        "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
        "role_manifest_inventory_sha256": contract.V38_ROLE_MANIFEST_INVENTORY_SHA256,
        "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
        "inventory_sha256": contract.V38_INVENTORY_SHA256,
    }
    expected_counts = {
        "documents": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "records": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "events": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "canonical_requests": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "pilots": contract.DEVELOPMENT_PILOT_COUNT,
        "remaining": contract.DEVELOPMENT_REMAINING_COUNT,
        "filename_present": contract.DEVELOPMENT_FILENAME_PRESENT_COUNT,
        "filename_missing": contract.DEVELOPMENT_FILENAME_MISSING_COUNT,
        "minimum_request_byte_count": counts["minimum_request_byte_count"],
        "maximum_request_byte_count": counts["maximum_request_byte_count"],
        "implementation_production_files": len(contract.IMPLEMENTATION_PRODUCTION_PATHS),
        "implementation_test_files": len(contract.IMPLEMENTATION_TEST_PATHS),
        "local_production_closure_files": len(
            contract.LOCAL_PRODUCTION_CLOSURE_PATHS
        ),
        "local_import_module_count": counts["local_import_module_count"],
        "experiment_family_sec_requests": contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
    }
    expected_privacy = {
        "readable_contact_published": False,
        "accessions_urls_filenames_offsets_bodies_published": False,
        "private_paths_published": False,
        "canonical_requests_published": False,
        "pre_release_science_values_published": False,
        "redacted_errors_only": True,
    }
    if (
        item["schema_version"] != PUBLIC_ARTIFACT_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "passed"
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["branch"] != contract.BRANCH_NAME
        or item["preregistration_commit"] != contract.PREREGISTRATION_COMMIT
        or item["preregistration_tree"] != contract.PREREGISTRATION_TREE
        or not _is_sha1(item["implementation_commit"])
        or not _is_sha1(item["implementation_tree"])
        or item["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or item["science_projection_sha256"] != contract.SCIENCE_PROJECTION_SHA256
        or not _is_sha256(item["runtime_binding_authority_sha256"])
        or not _is_sha256(item["private_manifest_sha256"])
        or not _is_sha256(item["private_manifest_literal_sha256"])
        or item["eligible_for_public_seal"] is not True
        or item["development_authorized"] is not False
        or source_authority != expected_source
        or type(counts["local_import_module_count"]) is not int
        or type(counts["minimum_request_byte_count"]) is not int
        or type(counts["maximum_request_byte_count"]) is not int
        or not (
            0
            < counts["minimum_request_byte_count"]
            <= counts["maximum_request_byte_count"]
            <= contract.MODEL_INPUT_MAX_BYTES
        )
        or not (
            1 + len(contract.IMPLEMENTATION_PRODUCTION_PATHS)
            <= counts["local_import_module_count"]
            <= len(contract.LOCAL_IMPORT_ALLOWED_PATHS)
        )
        or counts != expected_counts
        or any(not _is_sha256(child) for child in aggregates.values())
        or aggregates["local_production_closure_manifest_sha256"]
        != contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
        or aggregates["local_production_closure_order_sha256"]
        != contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
        or privacy != expected_privacy
        or any(child is not True for child in gates.values())
    ):
        _reject("preflight_public_artifact_invalid")
    _validate_effect_snapshot(item["effect_counts"])
    validate_qualification_report(item["qualification"])
    return item


def validate_public_failed_preflight_artifact(value: Any) -> dict[str, Any]:
    """Validate the fixed redacted terminal receipt for a consumed failure."""

    item = _strict_mapping(
        value,
        _PUBLIC_FAILED_PREFLIGHT_FIELDS,
        code="preflight_public_failure_shape_invalid",
    )
    _validate_self_hash(
        item,
        "public_artifact_sha256",
        code="preflight_public_failure_hash_invalid",
    )
    if (
        item["schema_version"] != PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "failed"
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["branch"] != contract.BRANCH_NAME
        or type(item["failure_code"]) is not str
        or re.fullmatch(r"[a-z0-9_]{3,80}", item["failure_code"]) is None
        or item["preflight_consumed"] is not True
        or item["failure_preserved"] is not True
        or item["redacted_error_only"] is not True
        or item["eligible_for_public_seal"] is not False
        or item["rerun_authorized"] is not False
        or item["development_authorized"] is not False
        or item["confirmation_and_final_opened"] is not False
        or item["real_money_authorized"] is not False
    ):
        _reject("preflight_public_failure_invalid")
    return item


def _recover_failed_preflight_publication(root: Path) -> None:
    """Finish only an exact failed-preflight pending promotion after a crash."""

    target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    pending = target.with_name(f"{target.name}.v315-pending")
    final_state = _optional_lstat(
        target, code="preflight_failure_preservation_failed"
    )
    pending_state = _optional_lstat(
        pending, code="preflight_failure_preservation_failed"
    )
    if pending_state is None:
        return
    if final_state is not None:
        _reject("preflight_failure_preservation_failed")

    _assert_regular_new_parent(target, root)
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    private_state = _optional_lstat(
        private_root, code="preflight_failure_preservation_failed"
    )
    if (
        private_state is None
        or not stat.S_ISDIR(private_state.st_mode)
        or stat.S_ISLNK(private_state.st_mode)
        or getattr(private_state, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        _reject("preflight_failure_preservation_failed")
    try:
        private_root.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _reject("preflight_failure_preservation_failed")
    _assert_exact_regular_file(
        private_root / PRIVATE_INTENT_FILENAME,
        _artifact_bytes(_preflight_intent()),
        code="preflight_failure_preservation_failed",
    )

    payload = _read_regular_file(
        pending,
        maximum=64 * 1024,
        code="preflight_failure_preservation_failed",
    )
    artifact = validate_public_failed_preflight_artifact(
        _parse_artifact_bytes(
            payload,
            maximum=64 * 1024,
            code="preflight_failure_preservation_failed",
        )
    )
    if _artifact_bytes(artifact) != payload:
        _reject("preflight_failure_preservation_failed")
    _write_atomic_new_file(
        target, payload, code="preflight_failure_preservation_failed"
    )


def _preserve_failed_preflight(
    root: Path,
    *,
    private_root: Path,
    failure_code: str,
    replace_exact_success: bytes | None = None,
) -> dict[str, Any]:
    """Publish one redacted failure after the private intent was consumed."""

    try:
        private_root.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _reject("preflight_failure_preservation_failed")
    body = {
        "schema_version": PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "failed",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "failure_code": failure_code,
        "preflight_consumed": True,
        "failure_preserved": True,
        "redacted_error_only": True,
        "eligible_for_public_seal": False,
        "rerun_authorized": False,
        "development_authorized": False,
        "confirmation_and_final_opened": False,
        "real_money_authorized": False,
    }
    artifact = validate_public_failed_preflight_artifact(
        _self_hash(body, "public_artifact_sha256")
    )
    payload = _artifact_bytes(artifact)
    safe_tokens = _privacy_token_bytes(
        root,
        lambda path: (
            str(path),
            str(path).replace("\\", "/"),
            str((path / V38_PRIVATE_ROOT).resolve(strict=False)),
            str(
                (path / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)
            ),
        ),
    )
    _scan_public_bytes(payload, safe_tokens)
    target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        _reject("preflight_failure_preservation_failed")
    _assert_regular_new_parent(target, root)
    if replace_exact_success is None:
        _write_atomic_new_file(
            target, payload, code="preflight_failure_preservation_failed"
        )
    else:
        _assert_exact_regular_file(
            target,
            replace_exact_success,
            code="preflight_failure_preservation_failed",
        )
        pending = target.with_name(f"{target.name}.v315-deadline-failure-pending")
        if _optional_lstat(
            pending, code="preflight_failure_preservation_failed"
        ) is not None:
            _reject("preflight_failure_preservation_failed")
        try:
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(target.parent)
            _assert_exact_regular_file(
                target,
                replace_exact_success,
                code="preflight_failure_preservation_failed",
            )
            os.replace(pending, target)
            _fsync_directory(target.parent)
        except V315PreflightError:
            raise
        except OSError:
            _reject("preflight_failure_preservation_failed")
        _assert_exact_regular_file(
            target, payload, code="preflight_failure_preservation_failed"
        )
    return artifact


def _repair_consumed_reservation_for_failure(
    root: Path,
    *,
    private_root: Path,
) -> None:
    """Best-effort completion of the intent after a partial reservation."""

    try:
        private_root.parent.mkdir(parents=True, exist_ok=True)
        private_root.mkdir(exist_ok=True)
        private_root.resolve(strict=True).relative_to(root)
        (private_root / PRIVATE_MANIFEST_DIRECTORY).mkdir(exist_ok=True)
        intent_path = private_root / PRIVATE_INTENT_FILENAME
        if _optional_lstat(
            intent_path, code="preflight_failure_preservation_failed"
        ) is None:
            _write_new_file(
                intent_path,
                _artifact_bytes(_preflight_intent()),
                code="preflight_failure_preservation_failed",
            )
    except V315PreflightError:
        raise
    except (OSError, ValueError):
        _reject("preflight_failure_preservation_failed")


def run_preflight(
    repo_root: Path,
    *,
    dependencies: PreflightDependencies | None = None,
) -> dict[str, Any]:
    """Consume and seal the v3.15 preflight exactly once, without external effects."""

    root = _canonical_repo_root(repo_root)
    pretrust = _consume_creation_pretrust_attestation(root)
    _recover_failed_preflight_publication(root)
    if (
        _optional_lstat(
            root / Path(contract.PREFLIGHT_ARTIFACT_PATH),
            code="preflight_already_consumed",
        )
        is not None
        or _optional_lstat(
            root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE),
            code="preflight_already_consumed",
        )
        is not None
    ):
        _reject("preflight_already_consumed")
    deps = dependencies if dependencies is not None else build_production_dependencies()
    if type(deps) is not PreflightDependencies:
        _reject("preflight_dependencies_invalid")
    # All operations through this point are read-only.  A failure is retryable
    # and must not leave a private or public byte.
    before_effects = _validate_effect_snapshot(deps.effect_snapshot())
    repository = validate_repository_snapshot(deps.inspect_repository(root))
    _validate_creation_pretrust_repository_binding(pretrust, repository)
    if before_effects != pretrust["effect_counts"]:
        _reject("creation_pretrust_effect_mismatch")

    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    consumption_started = False
    late_public_success: bytes | None = None

    def mark_consumed(observed: Path) -> None:
        nonlocal consumption_started
        if observed != private_root or consumption_started:
            _reject("preflight_reservation_invalid")
        consumption_started = True

    try:
        private_root = _reserve_once(root, on_consumption=mark_consumed)
        try:
            preflight_deadline = (
                time.monotonic() + contract.QUALIFICATION_TIMEOUT_SECONDS
            )
        except Exception:
            _reject("preflight_qualification_clock_invalid")
        if type(preflight_deadline) is not float or preflight_deadline <= 0.0:
            _reject("preflight_qualification_clock_invalid")
        _runtime_active_preflight_context()[
            "deadline_monotonic"
        ] = preflight_deadline
        # Qualification runs first.  Runtime authority is then created from I;
        # neither step is allowed to load or consume a future pushed F authority.
        tests = validate_qualification_report(deps.run_qualification(root))
        _runtime_require_deadline()
        _validate_qualification_private_state(
            private_root / "qualification",
            tests,
            repository=repository,
        )
        _runtime_require_deadline()
        runtime_authority = validate_runtime_binding_authority(
            deps.create_runtime_authority(root, repository)
        )
        _runtime_require_deadline()
        authority = deps.authenticate_source(root)
        _runtime_require_deadline()
        projection = deps.build_projection(authority)
        bridge = _projection_manifest(projection)
        commitments = validate_request_commitments(
            deps.build_request_commitments(projection)
        )
        _runtime_require_deadline()
        _validate_bridge_commitment_parity(bridge, commitments)
        after_effects = _validate_effect_snapshot(deps.effect_snapshot())
        if before_effects != after_effects:
            _reject("preflight_effect_accounting_changed")
        tokens = _privacy_token_bytes(root, deps.privacy_tokens)
        private_tree_tokens = _private_tree_forbidden_tokens(root, tokens)
        _scan_qualification_private_tree(
            root,
            forbidden=private_tree_tokens,
            code="preflight_private_privacy_scan_failed",
        )
        _runtime_require_deadline()

        private_manifest = _build_private_manifest(
            repository=repository,
            bridge=bridge,
            commitments=commitments,
            effect_before=before_effects,
            effect_after=after_effects,
            tests=tests,
            runtime_authority=runtime_authority,
        )
        validated_private = validate_private_preflight_manifest(private_manifest)
        private_payload = _artifact_bytes(validated_private)
        private_privacy = _scan_private_bytes(private_payload, tokens)
        if not all(private_privacy.values()):
            _reject("preflight_private_privacy_scan_failed")
        private_literal_sha256 = hashlib.sha256(private_payload).hexdigest()
        private_target = (
            private_root
            / PRIVATE_MANIFEST_DIRECTORY
            / f"{private_literal_sha256}.json"
        )
        _write_new_file(
            private_target,
            private_payload,
            code="preflight_private_manifest_unwritable",
        )
        _runtime_require_deadline()

        public_artifact = _build_public_artifact(
            repository=repository,
            bridge=bridge,
            commitments=commitments,
            effects=after_effects,
            tests=tests,
            runtime_authority=runtime_authority,
            private_manifest_sha256=validated_private["private_manifest_sha256"],
            private_manifest_literal_sha256=private_literal_sha256,
        )
        validated_public = validate_public_preflight_artifact(public_artifact)
        public_payload = _artifact_bytes(validated_public)
        public_privacy = _scan_public_bytes(public_payload, tokens)
        if not all(public_privacy.values()):
            _reject("preflight_public_privacy_scan_failed")
        _runtime_require_deadline()
        # Seal private success before the public file.  The public write is the
        # final fallible success step, so a preceding failure can still publish
        # the one allowed redacted failed artifact at that path.
        completion = _self_hash(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
                "private_manifest_sha256": validated_private[
                    "private_manifest_sha256"
                ],
                "private_manifest_literal_sha256": private_literal_sha256,
                "public_artifact_sha256": validated_public[
                    "public_artifact_sha256"
                ],
                "public_artifact_literal_sha256": hashlib.sha256(
                    public_payload
                ).hexdigest(),
                "status": "passed",
            },
            "completion_sha256",
        )
        _write_new_file(
            private_root / PRIVATE_COMPLETION_FILENAME,
            _artifact_bytes(completion),
            code="preflight_completion_unwritable",
        )
        _runtime_require_deadline()
        public_target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
        try:
            public_target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _reject("preflight_public_artifact_unwritable")
        _assert_regular_new_parent(public_target, root)
        _write_new_file(
            public_target,
            public_payload,
            code="preflight_public_artifact_unwritable",
        )
        late_public_success = public_payload
        _runtime_require_deadline()
        return validated_public
    except V315PreflightError as exc:
        if not consumption_started:
            raise
        failure_code = exc.code
    except Exception as exc:
        if not consumption_started:
            raise V315PreflightError("preflight_dependency_failed") from exc
        try:
            dependency_code = getattr(exc, "code", None)
        except Exception:
            dependency_code = None
        failure_code = (
            dependency_code
            if type(dependency_code) is str
            and re.fullmatch(r"[a-z0-9_]{3,80}", dependency_code) is not None
            else "preflight_dependency_failed"
        )
    _repair_consumed_reservation_for_failure(root, private_root=private_root)
    _preserve_failed_preflight(
        root,
        private_root=private_root,
        failure_code=failure_code,
        replace_exact_success=(
            late_public_success
            if failure_code == "preflight_qualification_timeout"
            else None
        ),
    )
    raise V315PreflightError(failure_code)


def _git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    trusted = _runtime_active_preflight_context().get("trusted_git")
    executable = (
        str(trusted[0])
        if isinstance(trusted, tuple)
        and len(trusted) == 3
        and isinstance(trusted[0], Path)
        else "git"
    )
    try:
        completed = subprocess.run(
            [executable, "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _reject("preflight_git_failed")
    if completed.returncode != 0:
        _reject("preflight_git_failed")
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        _reject("preflight_git_failed")


def _git_tree_rows(root: Path, revision: str) -> list[dict[str, str]]:
    text = _git(root, "ls-tree", "-r", "--full-tree", revision)
    if type(text) is not str:
        _reject("preflight_git_failed")
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        try:
            left, path = line.split("\t", 1)
            mode, kind, object_id = left.split(" ", 2)
        except ValueError:
            _reject("preflight_git_failed")
        if (
            not path
            or "\n" in path
            or "\r" in path
            or not _is_sha1(object_id)
        ):
            _reject("preflight_git_failed")
        rows.append(
            {"path": path, "mode": mode, "kind": kind, "object_id": object_id}
        )
    return rows


def _derive_local_production_closure(
    root: Path,
    *,
    head_commit: str,
) -> dict[str, Any]:
    """Reproduce the preregistered AST closure from immutable base blobs."""

    observed_tree = _git(root, "rev-parse", f"{contract.BASE_COMMIT}^{{tree}}")
    candidates_text = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        contract.BASE_COMMIT,
        "--",
        "agent_benchmark",
    )
    if (
        type(observed_tree) is not str
        or observed_tree != contract.BASE_TREE
        or type(candidates_text) is not str
    ):
        _reject("preflight_local_closure_base_invalid")
    candidates = {
        line
        for line in candidates_text.splitlines()
        if line.startswith("agent_benchmark/") and line.endswith(".py")
    }
    pending = [
        *contract.LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS,
        *contract.LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS,
    ]
    seen: set[str] = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        if path not in candidates:
            _reject("preflight_local_closure_missing")
        source = _git(root, "show", f"{contract.BASE_COMMIT}:{path}", binary=True)
        if type(source) is not bytes:
            _reject("preflight_local_closure_unreadable")
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError):
            _reject("preflight_local_closure_ast_invalid")
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent_benchmark."):
                        dependencies.add(alias.name.replace(".", "/") + ".py")
            elif isinstance(node, ast.ImportFrom):
                if (
                    node.level == 0
                    and node.module
                    and node.module.startswith("agent_benchmark.")
                ):
                    dependencies.add(node.module.replace(".", "/") + ".py")
                elif node.level == 1:
                    parent = path.rsplit("/", 1)[0]
                    if node.module:
                        dependencies.add(
                            parent + "/" + node.module.replace(".", "/") + ".py"
                        )
                    else:
                        for alias in node.names:
                            dependencies.add(
                                parent + "/" + alias.name.replace(".", "/") + ".py"
                            )
        seen.add(path)
        for dependency in sorted(
            dependencies & candidates,
            key=lambda value: value.encode("utf-8"),
        ):
            if dependency not in seen and dependency not in pending:
                pending.append(dependency)
    ordered = sorted(seen, key=lambda value: value.encode("utf-8"))
    if tuple(ordered) != contract.LOCAL_PRODUCTION_CLOSURE_PATHS:
        _reject("preflight_local_closure_paths_invalid")
    rows: list[dict[str, str]] = []
    for path in ordered:
        source = _git(root, "show", f"{contract.BASE_COMMIT}:{path}", binary=True)
        base_blob = _git(root, "rev-parse", f"{contract.BASE_COMMIT}:{path}")
        head_blob = _git(root, "rev-parse", f"{head_commit}:{path}")
        if (
            type(source) is not bytes
            or type(base_blob) is not str
            or type(head_blob) is not str
            or head_blob != base_blob
        ):
            _reject("preflight_local_closure_predecessor_changed")
        rows.append(
            {
                "git_blob_sha1": base_blob,
                "literal_sha256": hashlib.sha256(source).hexdigest(),
                "path": path,
            }
        )
    initializer = contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH
    base_initializer = _git(
        root, "rev-parse", f"{contract.BASE_COMMIT}:{initializer}"
    )
    head_initializer = _git(root, "rev-parse", f"{head_commit}:{initializer}")
    if (
        type(base_initializer) is not str
        or type(head_initializer) is not str
        or base_initializer != head_initializer
    ):
        _reject("preflight_package_initializer_changed")
    manifest = {
        "base_commit": contract.BASE_COMMIT,
        "base_tree": contract.BASE_TREE,
        "closure_path_count": len(ordered),
        "derivation": contract.LOCAL_PRODUCTION_CLOSURE_DERIVATION,
        "explicit_dynamic_local_paths": list(
            contract.LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS
        ),
        "ordered_roots": list(contract.LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS),
        "paths": rows,
        "schema": contract.LOCAL_PRODUCTION_CLOSURE_SCHEMA_VERSION,
    }
    return validate_local_production_closure_manifest(manifest)


def _audit_v315_local_import_upper_bound(
    root: Path,
    *,
    head_commit: str,
    runtime_observed_paths: Sequence[str] | None = None,
) -> list[str]:
    """Bind all six V3.15 modules to the strict local-import upper bound."""

    candidates_text = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        head_commit,
        "--",
        "agent_benchmark",
    )
    if type(candidates_text) is not str:
        _reject("preflight_local_import_upper_bound_invalid")
    candidates = {
        path
        for path in candidates_text.splitlines()
        if path.startswith("agent_benchmark/") and path.endswith(".py")
    }
    observed: set[str] = {
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
    }

    def module_path(name: str) -> str:
        if name == "agent_benchmark":
            return contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH
        return name.replace(".", "/") + ".py"

    for path in contract.IMPLEMENTATION_PRODUCTION_PATHS:
        source = _git(root, "show", f"{head_commit}:{path}", binary=True)
        if type(source) is not bytes:
            _reject("preflight_local_import_upper_bound_invalid")
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, TypeError, ValueError):
            _reject("preflight_local_import_upper_bound_invalid")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "agent_benchmark" or alias.name.startswith(
                        "agent_benchmark."
                    ):
                        observed.add(module_path(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == "agent_benchmark":
                    observed.add(contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH)
                    for alias in node.names:
                        candidate = module_path(
                            "agent_benchmark." + alias.name
                        )
                        if candidate in candidates:
                            observed.add(candidate)
                elif (
                    node.level == 0
                    and node.module
                    and node.module.startswith("agent_benchmark.")
                ):
                    observed.add(module_path(node.module))
                elif node.level == 1:
                    parent = path.rsplit("/", 1)[0]
                    if node.module:
                        observed.add(
                            parent + "/" + node.module.replace(".", "/") + ".py"
                        )
                    else:
                        for alias in node.names:
                            observed.add(
                                parent + "/" + alias.name.replace(".", "/") + ".py"
                            )
            elif isinstance(node, ast.Call):
                function_name: str | None = None
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                if function_name in {"__import__", "import_module", "run_module"}:
                    if not node.args:
                        _reject("preflight_local_import_dynamic_unresolved")
                    first = node.args[0]
                    if (
                        not isinstance(first, ast.Constant)
                        or type(first.value) is not str
                    ):
                        _reject("preflight_local_import_dynamic_unresolved")
                    if (
                        first.value == "agent_benchmark"
                        or first.value.startswith("agent_benchmark.")
                    ):
                        observed.add(module_path(first.value))
    if runtime_observed_paths is None:
        try:
            for module in tuple(sys.modules.values()):
                origin = getattr(getattr(module, "__spec__", None), "origin", None)
                if type(origin) is not str or origin in {"built-in", "frozen"}:
                    continue
                try:
                    relative = Path(origin).resolve(strict=True).relative_to(root)
                except (OSError, ValueError):
                    continue
                relative_text = relative.as_posix()
                if relative_text.startswith("agent_benchmark/"):
                    if relative_text.endswith(".py"):
                        observed.add(relative_text)
                    elif relative_text.endswith((".pyc", ".pyo")):
                        source_candidate = relative_text.rsplit(".", 1)[0] + ".py"
                        if source_candidate in candidates:
                            observed.add(source_candidate)
        except Exception:
            _reject("preflight_local_import_runtime_observation_failed")
    else:
        if (
            not isinstance(runtime_observed_paths, Sequence)
            or isinstance(runtime_observed_paths, (str, bytes, bytearray))
            or any(type(path) is not str for path in runtime_observed_paths)
        ):
            _reject("preflight_local_import_runtime_observation_failed")
        observed.update(runtime_observed_paths)
    ordered = sorted(observed, key=lambda value: value.encode("utf-8"))
    try:
        validated = contract.validate_local_import_paths(ordered)
    except contract.ContractViolation:
        _reject("preflight_local_import_upper_bound_invalid")
    return list(validated)


def inspect_pushed_implementation(repo_root: Path) -> dict[str, Any]:
    """Inspect local public Git state without fetching or changing it."""

    root = _canonical_repo_root(repo_root)
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    parent = _git_single_parent(
        root,
        commit,
        code="preflight_implementation_ancestry_invalid",
    )
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    changed_text = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        commit,
    )
    if not all(type(item) is str for item in (branch, commit, tree, parent, remote, status, changed_text)):
        _reject("preflight_git_failed")
    changed: dict[str, str] = {}
    for line in changed_text.splitlines():
        try:
            state, path = line.split("\t", 1)
        except ValueError:
            _reject("preflight_git_delta_invalid")
        if path in changed:
            _reject("preflight_git_delta_invalid")
        changed[path] = state
    ancestry = {
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "local_head": commit,
        "remote_head": remote,
        "changed_paths": changed,
        "clean_worktree": status == "",
        "preregistration_authenticated": True,
        "predecessor_blobs_unchanged": True,
    }
    try:
        contract.validate_implementation_ancestry(ancestry)
    except contract.ContractViolation:
        _reject("preflight_implementation_ancestry_invalid")

    preregistration = _git(
        root,
        "show",
        f"{contract.PREREGISTRATION_COMMIT}:{contract.PREREGISTRATION_PATH}",
        binary=True,
    )
    prereg_blob = _git(
        root,
        "rev-parse",
        f"{contract.PREREGISTRATION_COMMIT}:{contract.PREREGISTRATION_PATH}",
    )
    if (
        type(preregistration) is not bytes
        or type(prereg_blob) is not str
        or prereg_blob != contract.PREREGISTRATION_GIT_BLOB_SHA1
        or len(preregistration) != contract.PREREGISTRATION_LITERAL_BYTES
        or hashlib.sha256(preregistration).hexdigest()
        != contract.PREREGISTRATION_LITERAL_SHA256
    ):
        _reject("preflight_preregistration_invalid")

    source_rows: list[dict[str, Any]] = []
    for path in contract.IMPLEMENTATION_ALLOWED_PATHS:
        blob = _git(root, "show", f"{commit}:{path}", binary=True)
        blob_sha1 = _git(root, "rev-parse", f"{commit}:{path}")
        if type(blob) is not bytes or type(blob_sha1) is not str:
            _reject("preflight_source_inventory_invalid")
        try:
            working = _read_regular_file(
                root / Path(path),
                maximum=None,
                code="preflight_source_inventory_invalid",
            )
        except OSError:
            _reject("preflight_source_inventory_invalid")
        if working != blob:
            _reject("preflight_source_inventory_invalid")
        source_rows.append(
            {
                "path": path,
                "git_blob_sha1": blob_sha1,
                "literal_sha256": hashlib.sha256(blob).hexdigest(),
                "byte_count": len(blob),
            }
        )
    parent_rows = _git_tree_rows(root, contract.PREREGISTRATION_COMMIT)
    head_rows = _git_tree_rows(root, commit)
    predecessor_paths = {row["path"] for row in parent_rows}
    unchanged_rows = [row for row in head_rows if row["path"] in predecessor_paths]
    predecessor_hash = contract.canonical_sha256(parent_rows)
    unchanged_hash = contract.canonical_sha256(unchanged_rows)
    _derive_local_production_closure(root, head_commit=commit)
    local_import_paths = _audit_v315_local_import_upper_bound(
        root,
        head_commit=commit,
    )
    snapshot = {
        "ancestry": ancestry,
        "source_inventory": source_rows,
        "source_inventory_sha256": contract.canonical_sha256(source_rows),
        "production_source_inventory_sha256": contract.canonical_sha256(
            source_rows[: len(contract.IMPLEMENTATION_PRODUCTION_PATHS)]
        ),
        "test_source_inventory_sha256": contract.canonical_sha256(
            source_rows[len(contract.IMPLEMENTATION_PRODUCTION_PATHS) :]
        ),
        "predecessor_inventory_sha256": predecessor_hash,
        "unchanged_predecessor_inventory_sha256": unchanged_hash,
        "local_production_closure_manifest_sha256": (
            contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
        ),
        "local_production_closure_paths_sha256": (
            contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
        ),
        "local_import_paths": local_import_paths,
        "local_import_paths_sha256": contract.canonical_sha256(
            local_import_paths
        ),
        "package_initializer_predecessor_equal": True,
        "preregistration_authenticated": True,
    }
    return validate_repository_snapshot(snapshot)


def _parse_artifact_bytes(payload: bytes, *, maximum: int, code: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _reject(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if type(key) is not str or key in result:
                _reject(code)
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _reject(code),
        )
    except V315PreflightError:
        raise
    except (TypeError, ValueError, UnicodeError):
        _reject(code)
    if not isinstance(value, Mapping) or _artifact_bytes(value) != payload:
        _reject(code)
    return copy.deepcopy(dict(value))


def load_private_contact(repo_root: Path) -> str:
    """Load the ignored SEC contact without logging, hashing, or persisting it."""

    return _load_private_contact(_canonical_repo_root(repo_root))


def _read_execution_private_state(
    root: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    try:
        unresolved_details = private_root.lstat()
        resolved = private_root.resolve(strict=True)
        resolved.relative_to(root)
        details = resolved.lstat()
        names = {item.name for item in resolved.iterdir()}
    except (OSError, ValueError):
        _reject("execution_preflight_private_state_invalid")
    if (
        not stat.S_ISDIR(unresolved_details.st_mode)
        or stat.S_ISLNK(unresolved_details.st_mode)
        or getattr(unresolved_details, "st_file_attributes", 0)
        & _REPARSE_ATTRIBUTE
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or names
        != {
            PRIVATE_INTENT_FILENAME,
            PRIVATE_MANIFEST_DIRECTORY,
            PRIVATE_COMPLETION_FILENAME,
            "qualification",
            PRIVATE_RUNTIME_DIRECTORY,
        }
    ):
        _reject("execution_preflight_private_state_invalid")

    intent_payload = _read_regular_file(
        resolved / PRIVATE_INTENT_FILENAME,
        maximum=64 * 1024,
        code="execution_preflight_private_state_invalid",
    )
    completion_payload = _read_regular_file(
        resolved / PRIVATE_COMPLETION_FILENAME,
        maximum=64 * 1024,
        code="execution_preflight_private_state_invalid",
    )
    try:
        manifest_names = {
            item.name
            for item in (resolved / PRIVATE_MANIFEST_DIRECTORY).iterdir()
        }
    except OSError:
        _reject("execution_preflight_private_state_invalid")
    manifest_entries = _qualification_exact_directory(
        resolved / PRIVATE_MANIFEST_DIRECTORY,
        manifest_names,
        code="execution_preflight_private_state_invalid",
    )
    manifests = list(manifest_entries.values())
    intent = _parse_artifact_bytes(
        intent_payload,
        maximum=64 * 1024,
        code="execution_preflight_intent_invalid",
    )
    if set(intent) != {
        "schema_version",
        "preflight_attempt_id",
        "contract_manifest_sha256",
        "preregistration_commit",
        "external_effects_authorized",
        "intent_sha256",
    }:
        _reject("execution_preflight_intent_invalid")
    _validate_self_hash(
        intent, "intent_sha256", code="execution_preflight_intent_invalid"
    )
    if (
        intent["schema_version"] != PREFLIGHT_INTENT_SCHEMA_VERSION
        or intent["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or intent["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or intent["preregistration_commit"] != contract.PREREGISTRATION_COMMIT
        or intent["external_effects_authorized"] is not False
    ):
        _reject("execution_preflight_intent_invalid")

    if len(manifests) != 1:
        _reject("execution_preflight_private_manifest_invalid")
    manifest_path = manifests[0]
    manifest_payload = _read_regular_file(
        manifest_path,
        maximum=4 * 1024 * 1024,
        code="execution_preflight_private_manifest_invalid",
    )
    manifest_literal = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_path.name != f"{manifest_literal}.json":
        _reject("execution_preflight_private_manifest_invalid")
    manifest = validate_private_preflight_manifest(
        _parse_artifact_bytes(
            manifest_payload,
            maximum=4 * 1024 * 1024,
            code="execution_preflight_private_manifest_invalid",
        )
    )
    runtime_bundle = _load_private_runtime_bundle(root)
    if (
        runtime_bundle["runtime_binding_authority"][
            "runtime_binding_authority_sha256"
        ]
        != manifest["runtime_binding_authority_sha256"]
    ):
        _reject("execution_preflight_runtime_authority_invalid")
    _validate_qualification_private_state(
        resolved / "qualification",
        manifest["qualification"],
        repository=manifest["repository"],
    )

    completion = _parse_artifact_bytes(
        completion_payload,
        maximum=64 * 1024,
        code="execution_preflight_completion_invalid",
    )
    if set(completion) != {
        "schema_version",
        "preflight_attempt_id",
        "private_manifest_sha256",
        "private_manifest_literal_sha256",
        "public_artifact_sha256",
        "public_artifact_literal_sha256",
        "status",
        "completion_sha256",
    }:
        _reject("execution_preflight_completion_invalid")
    _validate_self_hash(
        completion,
        "completion_sha256",
        code="execution_preflight_completion_invalid",
    )
    if (
        completion["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or completion["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or completion["private_manifest_sha256"]
        != manifest["private_manifest_sha256"]
        or completion["private_manifest_literal_sha256"] != manifest_literal
        or not _is_sha256(completion["public_artifact_sha256"])
        or not _is_sha256(completion["public_artifact_literal_sha256"])
        or completion["status"] != "passed"
    ):
        _reject("execution_preflight_completion_invalid")
    return manifest, manifest_payload, completion, completion_payload


def _changed_paths(root: Path, parent: str, child: str) -> dict[str, str]:
    changed_text = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        child,
    )
    if type(changed_text) is not str:
        _reject("execution_preflight_git_invalid")
    changed: dict[str, str] = {}
    for line in changed_text.splitlines():
        try:
            state, path = line.split("\t", 1)
        except ValueError:
            _reject("execution_preflight_git_invalid")
        if path in changed:
            _reject("execution_preflight_git_invalid")
        changed[path] = state
    return changed


def _parse_canonical_mapping_bytes(
    payload: bytes, *, maximum: int, code: str
) -> dict[str, Any]:
    """Parse duplicate-free canonical JSON that has no trailing newline."""

    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _reject(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if type(key) is not str or key in result:
                _reject(code)
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _reject(code),
        )
    except V315PreflightError:
        raise
    except (TypeError, ValueError, UnicodeError):
        _reject(code)
    if not isinstance(value, Mapping):
        _reject(code)
    try:
        encoded = contract.canonical_json_bytes(value)
    except Exception:
        _reject(code)
    if encoded != payload:
        _reject(code)
    return copy.deepcopy(dict(value))


def _qualification_exact_directory(
    path: Path,
    expected_names: set[str],
    *,
    code: str,
) -> dict[str, Path]:
    """Return one exact ordinary directory inventory without following links."""

    try:
        details = path.lstat()
        children = list(path.iterdir())
    except OSError:
        _reject(code)
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or {child.name for child in children} != expected_names
        or len(children) != len(expected_names)
    ):
        _reject(code)
    return {child.name: child for child in children}


def _load_private_runtime_bundle(root: Path) -> dict[str, Any]:
    """Replay the exact marker-last runtime-authority bundle."""

    code = "runtime_authority_bundle_invalid"
    runtime_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE) / (
        PRIVATE_RUNTIME_DIRECTORY
    )
    entries = _qualification_exact_directory(
        runtime_root,
        {
            PRIVATE_RUNTIME_REPOSITORY_FILENAME,
            PRIVATE_RUNTIME_DEPENDENCY_FILENAME,
            PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY,
            PRIVATE_RUNTIME_AUTHORITY_FILENAME,
        },
        code=code,
    )

    def read_manifest(path: Path, hash_field: str) -> dict[str, Any]:
        value, _payload = _qualification_read_canonical(
            path, maximum=32 * 1024 * 1024, code=code
        )
        _validate_self_hash(value, hash_field, code=code)
        return value

    repository = validate_repository_runtime_manifest(
        read_manifest(
            entries[PRIVATE_RUNTIME_REPOSITORY_FILENAME],
            "repository_runtime_manifest_sha256",
        )
    )
    dependency = validate_execution_dependency_manifest(
        read_manifest(
            entries[PRIVATE_RUNTIME_DEPENDENCY_FILENAME],
            "execution_dependency_manifest_sha256",
        )
    )
    loaded_entries = _qualification_exact_directory(
        entries[PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY],
        {f"{name}.json" for name in RUNTIME_CHECKPOINT_NAMES},
        code=code,
    )
    loaded: dict[str, dict[str, Any]] = {}
    for checkpoint in RUNTIME_CHECKPOINT_NAMES:
        loaded[checkpoint] = validate_loaded_code_manifest(
            read_manifest(
                loaded_entries[f"{checkpoint}.json"],
                "loaded_code_manifest_sha256",
            )
        )
    authority_value, _authority_payload = _qualification_read_canonical(
        entries[PRIVATE_RUNTIME_AUTHORITY_FILENAME],
        maximum=4 * 1024 * 1024,
        code=code,
    )
    authority = validate_runtime_binding_authority(authority_value)
    if (
        authority["repository_runtime_manifest_sha256"]
        != repository["repository_runtime_manifest_sha256"]
        or authority["execution_dependency_manifest_sha256"]
        != dependency["execution_dependency_manifest_sha256"]
        or authority["checkpoint_loaded_code_manifest_sha256s"]
        != {
            name: loaded[name]["loaded_code_manifest_sha256"]
            for name in RUNTIME_CHECKPOINT_NAMES
        }
        or any(
            manifest["checkpoint_name"] != name
            or manifest["repository_runtime_manifest_sha256"]
            != repository["repository_runtime_manifest_sha256"]
            or manifest["execution_dependency_manifest_sha256"]
            != dependency["execution_dependency_manifest_sha256"]
            for name, manifest in loaded.items()
        )
        or dependency["sys_path_sha256"] != authority["final_sys_path_sha256"]
        or dependency["python_runtime"]["process_environment_sha256"]
        != authority["process_environment_sha256"]
    ):
        _reject(code)
    return {
        "repository_runtime_manifest": repository,
        "execution_dependency_manifest": dependency,
        "loaded_code_manifests": loaded,
        "runtime_binding_authority": authority,
    }


def _consume_runtime_launch_proof(
    root: Path, *, allowed_invocations: set[str]
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "invocation_kind",
            "repository_root_sha256",
            "bootstrap_sha256",
            "bootstrap_bytes",
            "process_environment_sha256",
            "final_sys_path_sha256",
            "runtime_launch_sha256",
        }
    )
    if not hasattr(builtins, SCIENTIFIC_RUNTIME_SENTINEL_NAME):
        _reject("runtime_launch_proof_missing")
    try:
        raw = getattr(builtins, SCIENTIFIC_RUNTIME_SENTINEL_NAME)
        delattr(builtins, SCIENTIFIC_RUNTIME_SENTINEL_NAME)
    except Exception:
        _reject("runtime_launch_proof_invalid")
    item = _strict_mapping(raw, fields, code="runtime_launch_proof_invalid")
    _validate_self_hash(
        item, "runtime_launch_sha256", code="runtime_launch_proof_invalid"
    )
    if (
        item["schema_version"]
        != "aapl-sec-gemma-lean-science-v3-15-runtime-launch-v1"
        or item["invocation_kind"] not in allowed_invocations
        or item["repository_root_sha256"]
        != private_windows_path_sha256(root)
        or item["bootstrap_sha256"] != SCIENTIFIC_BOOTSTRAP_SHA256
        or item["bootstrap_bytes"] != len(SCIENTIFIC_BOOTSTRAP_BYTES)
        or not _is_sha256(item["process_environment_sha256"])
        or not _is_sha256(item["final_sys_path_sha256"])
    ):
        _reject("runtime_launch_proof_invalid")
    return item


def _initialize_runtime_consumption_session(
    root: Path,
    *,
    allowed_invocations: set[str],
) -> dict[str, Any]:
    """Authenticate pushed expectations, then establish a fresh identity baseline."""

    key = private_windows_path_sha256(root)
    sessions = _runtime_binding_sessions()
    existing = sessions.get(key)
    if existing is not None:
        if (
            existing.get("mode") != "consumption"
            or existing.get("invocation_kind") not in allowed_invocations
        ):
            _reject("runtime_binding_session_invalid")
        return existing
    launch = _consume_runtime_launch_proof(
        root, allowed_invocations=allowed_invocations
    )
    bundle = _load_private_runtime_bundle(root)
    expected_repository = bundle["repository_runtime_manifest"]
    expected_dependency = bundle["execution_dependency_manifest"]
    expected_authority = bundle["runtime_binding_authority"]
    if (
        launch["process_environment_sha256"]
        != expected_authority["process_environment_sha256"]
        or launch["final_sys_path_sha256"]
        != expected_authority["final_sys_path_sha256"]
    ):
        _reject("runtime_launch_authority_mismatch")
    preload = _runtime_preload_repository_modules(root)
    current_repository = validate_repository_runtime_manifest(
        build_repository_runtime_manifest(
            root,
            implementation_commit=expected_repository["implementation_commit"],
            implementation_tree=expected_repository["implementation_tree"],
            origin_url=expected_repository["origin_url"],
        )
    )
    if current_repository != expected_repository:
        _reject("runtime_repository_binding_mismatch")
    current_dependency = validate_execution_dependency_manifest(
        _build_execution_dependency_manifest(
            root,
            repository_manifest=current_repository,
            preload=preload,
        )
    )
    if current_dependency != expected_dependency:
        _reject("runtime_dependency_binding_mismatch")
    critical = _runtime_live_critical_objects()
    first, baseline = _build_loaded_code_manifest(
        checkpoint_name=RUNTIME_CHECKPOINT_NAMES[0],
        repository_manifest=current_repository,
        dependency_manifest=current_dependency,
        critical_objects=critical,
    )
    first = validate_loaded_code_manifest(first)
    for checkpoint in RUNTIME_CHECKPOINT_NAMES:
        if checkpoint == RUNTIME_CHECKPOINT_NAMES[0]:
            current = first
        else:
            body = {
                name: copy.deepcopy(value)
                for name, value in first.items()
                if name != "loaded_code_manifest_sha256"
            }
            body["checkpoint_name"] = checkpoint
            current = {
                **body,
                "loaded_code_manifest_sha256": contract.canonical_sha256(body),
            }
        if current != bundle["loaded_code_manifests"][checkpoint]:
            _reject("runtime_loaded_code_binding_mismatch")
    session = {
        "mode": "consumption",
        "invocation_kind": launch["invocation_kind"],
        "authority_sha256": expected_authority[
            "runtime_binding_authority_sha256"
        ],
        "bundle": bundle,
        "baseline": baseline,
    }
    sessions[key] = session
    return session


def _runtime_porcelain_entry_count(payload: bytes) -> int:
    if type(payload) is not bytes:
        _reject("runtime_repository_state_invalid")
    if not payload:
        return 0
    if not payload.endswith(b"\x00"):
        _reject("runtime_repository_state_invalid")
    records = payload[:-1].split(b"\x00")
    count = 0
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            _reject("runtime_repository_state_invalid")
        status_pair = record[:2]
        if any(byte < 32 or byte > 126 for byte in status_pair):
            _reject("runtime_repository_state_invalid")
        index += 1
        if b"R" in status_pair or b"C" in status_pair:
            if index >= len(records) or not records[index]:
                _reject("runtime_repository_state_invalid")
            index += 1
        count += 1
    return count


def _runtime_repository_state(root: Path) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    if type(head) is not str or type(tree) is not str or type(status) is not bytes:
        _reject("runtime_repository_state_invalid")
    paths = sorted(
        (
            contract.PAUSE_ARTIFACT_PATH,
            f"{contract.PAUSE_ARTIFACT_PATH}.v315-pending",
            contract.RESULT_ARTIFACT_PATH,
            f"{contract.RESULT_ARTIFACT_PATH}.v315-pending",
            contract.COMPARISON_PATH,
            f"{contract.COMPARISON_PATH}.v315-pending",
        ),
        key=lambda value: value.encode("utf-8"),
    )
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = root / Path(relative)
        details = _optional_lstat(path, code="runtime_repository_state_invalid")
        if details is None:
            rows.append(
                {
                    "relative_path": relative,
                    "state": "absent",
                    "byte_count": None,
                    "literal_sha256": None,
                }
            )
        else:
            payload = _read_regular_file(
                path, maximum=None, code="runtime_repository_state_invalid"
            )
            rows.append(
                {
                    "relative_path": relative,
                    "state": "present",
                    "byte_count": len(payload),
                    "literal_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    body = {
        "schema_version": REPOSITORY_STATE_SCHEMA_VERSION,
        "head": head,
        "tree": tree,
        "git_status_porcelain_z_sha256": hashlib.sha256(status).hexdigest(),
        "git_status_entry_count": _runtime_porcelain_entry_count(status),
        "paths": rows,
    }
    return {**body, "repository_state_sha256": contract.canonical_sha256(body)}


def _runtime_validate_git_topology(
    root: Path,
    *,
    invocation_kind: str,
    authority: Mapping[str, Any],
) -> tuple[str, str]:
    branch = _git(root, "branch", "--show-current")
    upstream = _git(
        root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    origin = _git(root, "remote", "get-url", "origin")
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    remote = _git(
        root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
    )
    if (
        branch != contract.BRANCH_NAME
        or upstream != f"origin/{contract.BRANCH_NAME}"
        or origin != CREATION_PRETRUST_EXPECTED_ORIGIN_URL
        or not _is_sha1(head)
        or not _is_sha1(tree)
        or head != remote
    ):
        _reject("runtime_git_topology_invalid")
    try:
        preflight_commit = authority["preflight"]["commit"]
        implementation_commit = authority["implementation"]["commit"]
    except (KeyError, TypeError):
        _reject("runtime_git_topology_invalid")
    if (
        not _is_sha1(preflight_commit)
        or not _is_sha1(implementation_commit)
        or _git_single_parent(
            root, preflight_commit, code="runtime_git_topology_invalid"
        )
        != implementation_commit
        or _git_single_parent(
            root, implementation_commit, code="runtime_git_topology_invalid"
        )
        != contract.PREREGISTRATION_COMMIT
        or _changed_paths(root, implementation_commit, preflight_commit)
        != {contract.PREFLIGHT_ARTIFACT_PATH: "A"}
    ):
        _reject("runtime_git_topology_invalid")
    if invocation_kind == "development":
        if head != preflight_commit:
            _reject("runtime_git_topology_invalid")
    elif invocation_kind == "continuation":
        pause = _git_single_parent(root, head, code="runtime_git_topology_invalid")
        if (
            _git_single_parent(root, pause, code="runtime_git_topology_invalid")
            != preflight_commit
            or _changed_paths(root, preflight_commit, pause)
            != {contract.PAUSE_ARTIFACT_PATH: "A"}
            or _changed_paths(root, pause, head)
            != {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}
        ):
            _reject("runtime_git_topology_invalid")
    elif invocation_kind == "publication_recovery":
        if head != preflight_commit:
            pause = _git_single_parent(
                root, head, code="runtime_git_topology_invalid"
            )
            if (
                _git_single_parent(
                    root, pause, code="runtime_git_topology_invalid"
                )
                != preflight_commit
                or _changed_paths(root, preflight_commit, pause)
                != {contract.PAUSE_ARTIFACT_PATH: "A"}
                or _changed_paths(root, pause, head)
                != {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}
            ):
                _reject("runtime_git_topology_invalid")
    else:
        _reject("runtime_git_topology_invalid")
    return head, tree


def _runtime_authority_hash_from_execution(value: Mapping[str, Any]) -> str:
    try:
        observed = value["preflight"]["runtime_binding_authority_sha256"]
    except (KeyError, TypeError):
        if "runtime_binding_authority_sha256" in value:
            observed = value["runtime_binding_authority_sha256"]
        else:
            _reject("runtime_execution_authority_invalid")
    if not _is_sha256(observed):
        _reject("runtime_execution_authority_invalid")
    return observed


def _runtime_require_identity_map(
    expected: Mapping[Any, Any], observed: Mapping[Any, Any]
) -> None:
    """Reject replacements using live objects, never a session-saved echo."""

    if (
        not isinstance(expected, Mapping)
        or not isinstance(observed, Mapping)
        or set(expected) != set(observed)
        or any(observed[name] is not value for name, value in expected.items())
    ):
        _reject("runtime_binding_identity_mismatch")


def verify_runtime_binding_checkpoint(
    root: Path,
    *,
    authority: Mapping[str, Any],
    checkpoint_name: str,
    invocation_kind: str,
) -> dict[str, Any]:
    """Compare current state only with pushed F315 expectations and baseline."""

    repo_root = _canonical_repo_root(root)
    if (
        checkpoint_name not in RUNTIME_CHECKPOINT_NAMES
        or invocation_kind not in RUNTIME_ROUTE_ARGUMENTS
        or not isinstance(authority, Mapping)
    ):
        _reject("runtime_binding_mismatch")
    key = private_windows_path_sha256(repo_root)
    session = _runtime_binding_sessions().get(key)
    if (
        session is None
        or session.get("mode") != "consumption"
        or session.get("invocation_kind") != invocation_kind
    ):
        _reject("runtime_binding_session_missing")
    bundle = session["bundle"]
    expected_authority = bundle["runtime_binding_authority"]
    if (
        _runtime_authority_hash_from_execution(authority)
        != expected_authority["runtime_binding_authority_sha256"]
        or session["authority_sha256"]
        != expected_authority["runtime_binding_authority_sha256"]
    ):
        _reject("runtime_binding_mismatch")
    head, tree = _runtime_validate_git_topology(
        repo_root, invocation_kind=invocation_kind, authority=authority
    )
    expected_repository = bundle["repository_runtime_manifest"]
    current_repository = validate_repository_runtime_manifest(
        build_repository_runtime_manifest(
            repo_root,
            implementation_commit=expected_repository["implementation_commit"],
            implementation_tree=expected_repository["implementation_tree"],
            origin_url=expected_repository["origin_url"],
        )
    )
    if current_repository != expected_repository:
        _reject("runtime_binding_mismatch")
    preload = {
        "timezone_rows": bundle["execution_dependency_manifest"]["timezone_files"],
    }
    current_dependency = validate_execution_dependency_manifest(
        _build_execution_dependency_manifest(
            repo_root,
            repository_manifest=current_repository,
            preload=preload,
        )
    )
    if current_dependency != bundle["execution_dependency_manifest"]:
        _reject("runtime_binding_mismatch")
    live_critical = _runtime_live_critical_objects()
    current_loaded, current_baseline = _build_loaded_code_manifest(
        checkpoint_name=checkpoint_name,
        repository_manifest=current_repository,
        dependency_manifest=current_dependency,
        critical_objects=live_critical,
    )
    if current_loaded != bundle["loaded_code_manifests"][checkpoint_name]:
        _reject("runtime_binding_mismatch")
    baseline = session["baseline"]
    for family in (
        "module_identity",
        "namespace_identity",
        "callable_identity",
        "critical_identity",
        "graph_identity",
        "stateful_identity",
        "weak_callback_identity",
        "weak_self_ref_identity",
        "relative_path_identity",
        "typing_alias_identity",
        "pseudo_alias_identity",
    ):
        expected = baseline[family]
        observed = current_baseline[family]
        _runtime_require_identity_map(expected, observed)
    if (
        current_baseline["owner_slot_arrays"] != baseline["owner_slot_arrays"]
        or
        current_baseline["module_names"] != baseline["module_names"]
        or current_baseline["binary_tokens"] != baseline["binary_tokens"]
    ):
        _reject("runtime_binding_identity_mismatch")
    state = _runtime_repository_state(repo_root)
    if invocation_kind == "publication_recovery":
        # The recovery route alone may be dirty, and only in the exact
        # idempotent pause/result pending shapes accepted by this validator.
        _publication_path_state(repo_root)
    elif state["git_status_entry_count"] != 0:
        _reject("runtime_binding_dirty_state")
    python_runtime = current_dependency["python_runtime"]
    interpreter = build_interpreter_identity(
        executable_sha256=python_runtime["executable_sha256"],
        python_dll_sha256=python_runtime["python_dll_sha256"],
        python_version=python_runtime["python_version"],
        cache_tag=python_runtime["cache_tag"],
        launcher_profile_sha256=python_runtime["launcher_profile_sha256"],
        flags_sha256=python_runtime["flags_sha256"],
        process_environment_sha256=python_runtime[
            "process_environment_sha256"
        ],
        final_sys_path_sha256=current_dependency["sys_path_sha256"],
    )
    return {
        "head": head,
        "tree": tree,
        "clean_state_sha256": state["repository_state_sha256"],
        "repository_runtime_manifest_sha256": current_repository[
            "repository_runtime_manifest_sha256"
        ],
        "execution_dependency_manifest_sha256": current_dependency[
            "execution_dependency_manifest_sha256"
        ],
        "loaded_code_manifest_sha256": current_loaded[
            "loaded_code_manifest_sha256"
        ],
        "runtime_binding_authority_sha256": expected_authority[
            "runtime_binding_authority_sha256"
        ],
        "route_argv_sha256": expected_authority["route_argv_sha256s"][
            invocation_kind
        ],
        "process_environment_sha256": expected_authority[
            "process_environment_sha256"
        ],
        "interpreter_identity_sha256": interpreter,
    }


def _qualification_read_canonical(
    path: Path,
    *,
    maximum: int,
    code: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_file(path, maximum=maximum, code=code)
    return (
        _parse_canonical_mapping_bytes(payload, maximum=maximum, code=code),
        payload,
    )


def _validate_qualification_runtime_receipt(
    value: Any,
    *,
    code: str,
) -> dict[str, Any]:
    runtime = _strict_mapping(value, _QUALIFICATION_RUNTIME_FIELDS, code=code)
    if (
        runtime["python_version"] != contract.QUALIFICATION_PYTHON_VERSION
        or runtime["python_cache_tag"]
        != contract.QUALIFICATION_PYTHON_CACHE_TAG
        or runtime["os_name"] != contract.QUALIFICATION_OS_NAME
        or runtime["sys_platform"] != contract.QUALIFICATION_SYS_PLATFORM
        or runtime["executable_basename"]
        != contract.QUALIFICATION_EXECUTABLE_BASENAME
        or runtime["executable_bytes"]
        != contract.QUALIFICATION_EXECUTABLE_BYTES
        or runtime["executable_sha256"]
        != contract.QUALIFICATION_EXECUTABLE_SHA256
        or runtime["pytest_version"] != contract.QUALIFICATION_PYTEST_VERSION
        or runtime["pytest_init_bytes"]
        != contract.QUALIFICATION_PYTEST_INIT_BYTES
        or runtime["pytest_init_sha256"]
        != contract.QUALIFICATION_PYTEST_INIT_SHA256
        or type(runtime["executable_path"]) is not str
        or not runtime["executable_path"]
        or not Path(runtime["executable_path"]).is_absolute()
        or Path(runtime["executable_path"]).name
        != contract.QUALIFICATION_EXECUTABLE_BASENAME
        or type(runtime["pytest_init_path"]) is not str
        or not runtime["pytest_init_path"]
        or not Path(runtime["pytest_init_path"]).is_absolute()
        or Path(runtime["pytest_init_path"]).name != "__init__.py"
    ):
        _reject(code)
    return runtime


def _qualification_validate_node_ids(
    value: Any,
    *,
    phase: str,
    count: Any,
    digest: Any,
    code: str,
) -> list[str]:
    if (
        type(value) is not list
        or not value
        or any(type(node_id) is not str or not node_id for node_id in value)
        or type(count) is not int
        or count != len(value)
        or not _is_sha256(digest)
    ):
        _reject(code)
    encoded = "\n".join(value).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != digest:
        _reject(code)
    multiplicities = Counter(value)
    duplicates = {
        node_id: observed
        for node_id, observed in multiplicities.items()
        if observed != 1
    }
    if phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
        if (
            len(value) != contract.QUALIFICATION_SHARED_NODE_COUNT
            or len(multiplicities)
            != contract.QUALIFICATION_SHARED_UNIQUE_NODE_COUNT
            or len(duplicates)
            != contract.QUALIFICATION_SHARED_DUPLICATE_NODE_COUNT
            or duplicates
        ):
            _reject(code)
    elif phase == contract.QUALIFICATION_PHASE_V315:
        if (
            len(value) < contract.QUALIFICATION_LATEST_MIN_NODE_COUNT
            or duplicates
            or any(
                multiplicities.get(node_id) != 1
                for node_id in _RUNTIME_REQUIRED_PHASE1_NODE_IDS
            )
        ):
            _reject(code)
    elif phase == contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY:
        if (
            duplicates
            or value != [contract.QUALIFICATION_SENTINEL_NODE_ID]
            or digest != contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
        ):
            _reject(code)
    else:
        _reject(code)
    return list(value)


def _validate_qualification_private_state(
    qualification_root: Path,
    report_value: Any,
    *,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay every immutable private qualification receipt fail closed."""

    code = "execution_preflight_qualification_invalid"
    report = validate_qualification_report(report_value)
    ancestry = validate_repository_snapshot(repository)["ancestry"]
    root_entries = _qualification_exact_directory(
        qualification_root,
        {
            *contract.QUALIFICATION_PHASES,
            "private-aggregate.json",
            "private-aggregate.complete.json",
        },
        code=code,
    )
    aggregate, aggregate_payload = _qualification_read_canonical(
        root_entries["private-aggregate.json"],
        maximum=32 * 1024 * 1024,
        code=code,
    )
    aggregate = _strict_mapping(
        aggregate,
        _QUALIFICATION_PRIVATE_AGGREGATE_FIELDS,
        code=code,
    )
    _validate_self_hash(
        aggregate,
        "private_aggregate_sha256",
        code=code,
    )
    runtime = _validate_qualification_runtime_receipt(
        aggregate["runtime"], code=code
    )
    if (
        aggregate["schema_version"]
        != contract.QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION
        or aggregate["suite_identity"] != contract.QUALIFICATION_SUITE_ID
        or aggregate["status"] != "passed"
        or type(aggregate["deadline_monotonic_ns"]) is not int
        or aggregate["deadline_monotonic_ns"] <= 0
        or aggregate["deadline_seconds"]
        != contract.QUALIFICATION_TIMEOUT_SECONDS
        or aggregate["durability_mode"]
        != contract.QUALIFICATION_DURABILITY_MODE
        or aggregate["repository_commit"] != ancestry["commit"]
        or aggregate["repository_tree"] != ancestry["tree"]
        or aggregate["clean_state_sha256"] != hashlib.sha256(b"").hexdigest()
        or aggregate["private_aggregate_sha256"]
        != report["private_aggregate_sha256"]
        or type(aggregate["phases"]) is not list
        or len(aggregate["phases"]) != len(contract.QUALIFICATION_PHASES)
    ):
        _reject(code)

    parsed_phase_rows: list[dict[str, Any]] = []
    derived_public_phases: list[dict[str, Any]] = []
    total_duration_ns = 0
    for expected_phase, public_phase, aggregate_phase_value in zip(
        contract.QUALIFICATION_PHASES,
        report["phases"],
        aggregate["phases"],
        strict=True,
    ):
        phase_entries = _qualification_exact_directory(
            root_entries[expected_phase],
            {"collection", "execution", "collection-manifest.json"},
            code=code,
        )
        collection_manifest, _collection_manifest_payload = (
            _qualification_read_canonical(
                phase_entries["collection-manifest.json"],
                maximum=8 * 1024 * 1024,
                code=code,
            )
        )
        collection_manifest = _strict_mapping(
            collection_manifest,
            _QUALIFICATION_COLLECTION_FIELDS,
            code=code,
        )
        _validate_self_hash(
            collection_manifest,
            "collection_manifest_sha256",
            code=code,
        )
        if (
            collection_manifest["schema_version"]
            != contract.QUALIFICATION_COLLECTION_SCHEMA_VERSION
            or collection_manifest["suite_identity"]
            != contract.QUALIFICATION_SUITE_ID
            or collection_manifest["phase"] != expected_phase
            or collection_manifest["status"] != "passed"
        ):
            _reject(code)
        sealed_node_ids = _qualification_validate_node_ids(
            collection_manifest["ordered_node_ids"],
            phase=expected_phase,
            count=collection_manifest["node_count"],
            digest=collection_manifest["node_list_sha256"],
            code=code,
        )
        selectors = _qualification_selectors(expected_phase)
        selector_files = {selector.split("::", 1)[0] for selector in selectors}
        if any(
            node_id.split("::", 1)[0] not in selector_files
            for node_id in sealed_node_ids
        ):
            _reject(code)
        if (
            expected_phase == contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY
            and sealed_node_ids != [contract.QUALIFICATION_SENTINEL_NODE_ID]
        ):
            _reject(code)

        parsed_modes: dict[str, dict[str, Any]] = {}
        parsed_completions: dict[str, dict[str, Any]] = {}
        for mode in contract.QUALIFICATION_MODES:
            mode_entries = _qualification_exact_directory(
                phase_entries[mode],
                {
                    "intent.json",
                    "output.log",
                    "stdout.bin",
                    "junit.xml",
                    "result.json",
                    "complete.json",
                },
                code=code,
            )
            intent, _intent_payload = _qualification_read_canonical(
                mode_entries["intent.json"], maximum=512 * 1024, code=code
            )
            intent = _strict_mapping(
                intent, _QUALIFICATION_INTENT_FIELDS, code=code
            )
            _validate_self_hash(intent, "intent_sha256", code=code)
            intent_runtime = _validate_qualification_runtime_receipt(
                intent["runtime"], code=code
            )
            junit_partial = qualification_root.parent / Path(
                contract.QUALIFICATION_JUNIT_RELATIVE_TEMPLATE.format(
                    phase=expected_phase,
                    mode=mode,
                )
            )
            repo_root = qualification_root.parent
            for _part in Path(contract.PRIVATE_PREFLIGHT_NAMESPACE).parts:
                repo_root = repo_root.parent
            expected_argv = _qualification_argv(
                repo_root,
                runtime=runtime,
                junit_partial=junit_partial,
                phase=expected_phase,
                mode=mode,
            )
            _expected_environment, expected_environment_profile = (
                _qualification_environment(runtime)
            )
            expected_collection_hash = (
                None
                if mode == "collection"
                else collection_manifest["collection_manifest_sha256"]
            )
            if (
                intent["schema_version"]
                != contract.QUALIFICATION_INTENT_SCHEMA_VERSION
                or intent["suite_identity"] != contract.QUALIFICATION_SUITE_ID
                or intent["phase"] != expected_phase
                or intent["mode"] != mode
                or intent["argv"] != expected_argv
                or intent["environment_profile"]
                != expected_environment_profile
                or intent["repository_commit"]
                != aggregate["repository_commit"]
                or intent["repository_tree"] != aggregate["repository_tree"]
                or intent["clean_state_sha256"]
                != aggregate["clean_state_sha256"]
                or intent["deadline_monotonic_ns"]
                != aggregate["deadline_monotonic_ns"]
                or intent_runtime != runtime
                or intent["collection_manifest_sha256"]
                != expected_collection_hash
            ):
                _reject(code)

            result, result_payload = _qualification_read_canonical(
                mode_entries["result.json"],
                maximum=16 * 1024 * 1024,
                code=code,
            )
            result = _strict_mapping(
                result, _QUALIFICATION_RESULT_FIELDS, code=code
            )
            _validate_self_hash(result, "result_sha256", code=code)
            result_node_ids = _qualification_validate_node_ids(
                result["node_ids"],
                phase=expected_phase,
                count=result["node_count"],
                digest=result["node_list_sha256"],
                code=code,
            )
            log_payload = _read_regular_file(
                mode_entries["output.log"],
                maximum=_QUALIFICATION_MAX_LOG_BYTES * 2,
                code=code,
            )
            stdout_payload = _read_regular_file(
                mode_entries["stdout.bin"],
                maximum=_QUALIFICATION_MAX_LOG_BYTES,
                code=code,
            )
            xml_payload = _read_regular_file(
                mode_entries["junit.xml"],
                maximum=_QUALIFICATION_MAX_XML_BYTES,
                code=code,
            )
            if (
                result["schema_version"]
                != contract.QUALIFICATION_RESULT_SCHEMA_VERSION
                or result["suite_identity"] != contract.QUALIFICATION_SUITE_ID
                or result["phase"] != expected_phase
                or result["mode"] != mode
                or result["status"] != "passed"
                or result["intent_sha256"] != intent["intent_sha256"]
                or result["collection_manifest_sha256"]
                != expected_collection_hash
                or type(result["started_unix_ns"]) is not int
                or result["started_unix_ns"] < 0
                or type(result["ended_unix_ns"]) is not int
                or result["ended_unix_ns"] < 0
                or result["ended_unix_ns"] < result["started_unix_ns"]
                or type(result["duration_monotonic_ns"]) is not int
                or result["duration_monotonic_ns"] < 0
                or result["exit_code"] != 0
                or result["timed_out"] is not False
                or result["deadline_overrun"] is not False
                or result["terminal_reason"] != "completed"
                or result["exception_code"] is not None
                or result["log_bytes"] != len(log_payload)
                or result["log_sha256"]
                != hashlib.sha256(log_payload).hexdigest()
                or result["stdout_bytes"] != len(stdout_payload)
                or result["stdout_sha256"]
                != hashlib.sha256(stdout_payload).hexdigest()
                or result["xml_present"] is not True
                or result["xml_bytes"] != len(xml_payload)
                or result["xml_sha256"]
                != hashlib.sha256(xml_payload).hexdigest()
            ):
                _reject(code)
            try:
                xml_root = ET.fromstring(xml_payload)
            except ET.ParseError:
                _reject(code)
            if mode == "collection":
                replay_node_ids = _qualification_collection_node_ids(
                    stdout_payload
                )
                replay_counts = _qualification_junit_counts(
                    xml_payload,
                    stdout_payload,
                    require_terminal_summary=False,
                )
                if (
                    result_node_ids != sealed_node_ids
                    or replay_node_ids != sealed_node_ids
                    or any(
                        replay_counts[field] != result[field]
                        for field in _QUALIFICATION_COUNT_FIELDS
                    )
                    or list(xml_root.iter("testcase"))
                    or any(result[field] != 0 for field in _QUALIFICATION_COUNT_FIELDS)
                ):
                    _reject(code)
            else:
                replay_count, replay_hash, replay_node_ids = (
                    _qualification_executed_node_evidence(
                        xml_payload, sealed_node_ids
                    )
                )
                replay_counts = _qualification_junit_counts(
                    xml_payload, stdout_payload
                )
                if (
                    replay_count != result["node_count"]
                    or replay_hash != result["node_list_sha256"]
                    or replay_node_ids != result_node_ids
                    or replay_node_ids != sealed_node_ids
                    or any(
                        replay_counts[field] != result[field]
                        for field in _QUALIFICATION_COUNT_FIELDS
                    )
                    or result["passed_count"] != result["node_count"]
                    or any(
                        result[field] != 0
                        for field in _QUALIFICATION_COUNT_FIELDS[1:]
                    )
                ):
                    _reject(code)

            completion, _completion_payload = _qualification_read_canonical(
                mode_entries["complete.json"], maximum=512 * 1024, code=code
            )
            completion = _strict_mapping(
                completion, _QUALIFICATION_COMPLETION_FIELDS, code=code
            )
            _validate_self_hash(completion, "completion_sha256", code=code)
            if (
                completion["schema_version"]
                != QUALIFICATION_COMPLETION_SCHEMA_VERSION
                or completion["suite_identity"]
                != contract.QUALIFICATION_SUITE_ID
                or completion["phase"] != expected_phase
                or completion["mode"] != mode
                or completion["status"] != "completed"
                or completion["terminal_reason"] != "completed"
                or completion["deadline_overrun"] is not False
                or completion["result_sha256"] != result["result_sha256"]
                or completion["result_literal_sha256"]
                != hashlib.sha256(result_payload).hexdigest()
                or completion["log_sha256"] != result["log_sha256"]
                or completion["stdout_sha256"] != result["stdout_sha256"]
                or completion["xml_sha256"] != result["xml_sha256"]
            ):
                _reject(code)
            parsed_modes[mode] = result
            parsed_completions[mode] = completion
            total_duration_ns += result["duration_monotonic_ns"]

        collection_result = parsed_modes["collection"]
        execution_result = parsed_modes["execution"]
        if (
            collection_manifest["collection_result_sha256"]
            != collection_result["result_sha256"]
            or collection_manifest["collection_completion_sha256"]
            != parsed_completions["collection"]["completion_sha256"]
            or collection_manifest["node_count"]
            != collection_result["node_count"]
            or collection_manifest["node_list_sha256"]
            != collection_result["node_list_sha256"]
            or execution_result["node_count"]
            != collection_manifest["node_count"]
            or execution_result["node_list_sha256"]
            != collection_manifest["node_list_sha256"]
        ):
            _reject(code)
        if expected_phase == contract.QUALIFICATION_PHASE_V315:
            if collection_manifest["node_count"] < (
                contract.QUALIFICATION_LATEST_MIN_NODE_COUNT
            ):
                _reject(code)
        elif expected_phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
            if (
                collection_manifest["node_count"]
                != contract.QUALIFICATION_SHARED_NODE_COUNT
                or collection_manifest["node_list_sha256"]
                != contract.QUALIFICATION_SHARED_NODE_LIST_SHA256
            ):
                _reject(code)
        elif (
            sealed_node_ids != [contract.QUALIFICATION_SENTINEL_NODE_ID]
            or collection_manifest["node_count"]
            != contract.QUALIFICATION_SENTINEL_NODE_COUNT
            or collection_manifest["node_list_sha256"]
            != contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
        ):
            _reject(code)

        aggregate_phase = _strict_mapping(
            aggregate_phase_value,
            _QUALIFICATION_PRIVATE_PHASE_FIELDS,
            code=code,
        )
        expected_aggregate_phase = {
            "phase": expected_phase,
            "node_count": collection_manifest["node_count"],
            "node_list_sha256": collection_manifest["node_list_sha256"],
            "collection_manifest_sha256": collection_manifest[
                "collection_manifest_sha256"
            ],
            "collection_completion_sha256": parsed_completions["collection"][
                "completion_sha256"
            ],
            "execution_completion_sha256": parsed_completions["execution"][
                "completion_sha256"
            ],
            "collection": collection_result,
            "execution": execution_result,
        }
        if aggregate_phase != expected_aggregate_phase:
            _reject(code)
        parsed_phase_rows.append(expected_aggregate_phase)
        derived_public_phases.append(
            {
                "phase": expected_phase,
                "node_count": collection_result["node_count"],
                "node_list_sha256": collection_result["node_list_sha256"],
                "collection_duration_ns": collection_result[
                    "duration_monotonic_ns"
                ],
                "execution_duration_ns": execution_result[
                    "duration_monotonic_ns"
                ],
                "collection_exit_code": collection_result["exit_code"],
                "execution_exit_code": execution_result["exit_code"],
                **{
                    field: execution_result[field]
                    for field in _QUALIFICATION_COUNT_FIELDS
                },
                "collection_result_sha256": collection_result[
                    "result_sha256"
                ],
                "execution_result_sha256": execution_result[
                    "result_sha256"
                ],
                "collection_log_sha256": collection_result["log_sha256"],
                "execution_log_sha256": execution_result["log_sha256"],
                "collection_xml_sha256": collection_result["xml_sha256"],
                "execution_xml_sha256": execution_result["xml_sha256"],
                "collection_manifest_sha256": collection_manifest[
                    "collection_manifest_sha256"
                ],
            }
        )
        if derived_public_phases[-1] != public_phase:
            _reject(code)

    if aggregate["phases"] != parsed_phase_rows:
        _reject(code)
    if total_duration_ns >= contract.QUALIFICATION_TIMEOUT_SECONDS * 1_000_000_000:
        _reject(code)
    derived_report = _self_hash(
        {
            "schema_version": QUALIFICATION_REPORT_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "status": "passed",
            "phase_count": len(derived_public_phases),
            "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
            "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
            "command_profile_sha256": QUALIFICATION_COMMAND_PROFILE_SHA256,
            "phases": derived_public_phases,
            "private_aggregate_sha256": aggregate[
                "private_aggregate_sha256"
            ],
        },
        "qualification_sha256",
    )
    if derived_report != report:
        _reject(code)

    aggregate_completion, _aggregate_completion_payload = (
        _qualification_read_canonical(
            root_entries["private-aggregate.complete.json"],
            maximum=512 * 1024,
            code=code,
        )
    )
    aggregate_completion = _strict_mapping(
        aggregate_completion,
        _QUALIFICATION_COMPLETION_FIELDS,
        code=code,
    )
    _validate_self_hash(
        aggregate_completion, "completion_sha256", code=code
    )
    if (
        aggregate_completion["schema_version"]
        != QUALIFICATION_COMPLETION_SCHEMA_VERSION
        or aggregate_completion["suite_identity"]
        != contract.QUALIFICATION_SUITE_ID
        or aggregate_completion["phase"] != "aggregate"
        or aggregate_completion["mode"] != "aggregate"
        or aggregate_completion["status"] != "completed"
        or aggregate_completion["terminal_reason"] != "completed"
        or aggregate_completion["deadline_overrun"] is not False
        or aggregate_completion["result_sha256"]
        != aggregate["private_aggregate_sha256"]
        or aggregate_completion["result_literal_sha256"]
        != hashlib.sha256(aggregate_payload).hexdigest()
        or aggregate_completion["log_sha256"] is not None
        or aggregate_completion["stdout_sha256"] is not None
        or aggregate_completion["xml_sha256"] is not None
    ):
        _reject(code)
    return aggregate


def _git_single_parent(root: Path, revision: str, *, code: str) -> str:
    value = _git(root, "rev-list", "--parents", "-n", "1", revision)
    if type(value) is not str:
        _reject(code)
    fields = value.split()
    if len(fields) != 2 or fields[0] != revision or not _is_sha1(fields[1]):
        _reject(code)
    return fields[1]


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _git_blob_material(
    root: Path, revision: str, path: str, *, maximum: int, code: str
) -> dict[str, Any]:
    payload = _git(root, "show", f"{revision}:{path}", binary=True)
    object_id = _git(root, "rev-parse", f"{revision}:{path}")
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > maximum
        or type(object_id) is not str
        or not _is_sha1(object_id)
        or _git_blob_sha1(payload) != object_id
    ):
        _reject(code)
    return {
        "path": path,
        "git_blob_sha1": object_id,
        "literal_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "payload": payload,
    }


def _inspect_result_git(root: Path) -> dict[str, Any]:
    """Authenticate the pushed R shell before any private value is loaded."""

    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    remote = _git(
        root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
    )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not all(type(item) is str for item in (branch, commit, remote, status, tree)):
        _reject("pushed_result_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or not _is_sha1(commit)
        or not _is_sha1(tree)
        or commit != remote
        or status != ""
    ):
        _reject("pushed_result_git_invalid")
    parent = _git_single_parent(root, commit, code="pushed_result_ancestry_invalid")
    changed = _changed_paths(root, parent, commit)
    if changed != {
        contract.RESULT_ARTIFACT_PATH: "A",
        contract.COMPARISON_PATH: "M",
    }:
        _reject("pushed_result_delta_invalid")
    result_blob = _git_blob_material(
        root,
        commit,
        contract.RESULT_ARTIFACT_PATH,
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_artifact_blob_invalid",
    )
    comparison_blob = _git_blob_material(
        root,
        commit,
        contract.COMPARISON_PATH,
        maximum=MAX_COMPARISON_BYTES,
        code="pushed_result_comparison_blob_invalid",
    )
    try:
        working_result = _read_regular_file(
            root / Path(contract.RESULT_ARTIFACT_PATH),
            maximum=MAX_RESULT_ARTIFACT_BYTES,
            code="pushed_result_worktree_invalid",
        )
        working_comparison = _read_regular_file(
            root / Path(contract.COMPARISON_PATH),
            maximum=MAX_COMPARISON_BYTES,
            code="pushed_result_worktree_invalid",
        )
    except OSError:
        _reject("pushed_result_worktree_invalid")
    if (
        working_result != result_blob["payload"]
        or working_comparison != comparison_blob["payload"]
    ):
        _reject("pushed_result_worktree_invalid")
    return {
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "remote": remote,
        "status": status,
        "changed_paths": changed,
        "result_blob": result_blob,
        "comparison_blob": comparison_blob,
    }


def _validate_replayed_pilot_guard(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUSHED_RESULT_PILOT_GUARD_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        item = contract.validate_self_sha256(item, field="segment_guard_sha256")
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    events = item["ordered_generation_response_event_sha256s"]
    if (
        item["schema_version"] != PUSHED_RESULT_RUNTIME_GUARD_SCHEMA_VERSION
        or item["segment_id"] != "pilot"
        or item["store_segment_id"] != "initial"
        or item["generation_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or not all(
            _is_sha256(item[field])
            for field in (
                "pre_runtime_receipt_sha256",
                "post_runtime_receipt_sha256",
                "stable_runtime_identity_sha256",
            )
        )
        or type(events) is not list
        or len(events) != contract.DEVELOPMENT_PILOT_COUNT
        or any(not _is_sha256(child) for child in events)
        or item["ordered_generation_response_events_sha256"]
        != contract.canonical_sha256(events)
        or item["raw_show_hash_is_diagnostic_only"] is not True
        or item["modified_at_is_excluded_only"] is not True
        or item["identity_http_request_count"] != 4
        or item["retry_count"] != 0
    ):
        _reject("pushed_result_pause_artifact_invalid")
    return item


def _validate_replayed_pilot_latency(value: Any, *, plan: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUSHED_RESULT_LATENCY_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        item = contract.validate_self_sha256(
            item, field="latency_receipt_sha256"
        )
        execution_calls = list(getattr(plan, "execution_calls"))
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    rows_raw = item["pilot_rows"]
    if (
        type(rows_raw) is not list
        or len(rows_raw) != contract.DEVELOPMENT_PILOT_COUNT
        or len(execution_calls) != contract.DEVELOPMENT_DOCUMENT_COUNT
    ):
        _reject("pushed_result_pause_artifact_invalid")
    rows: list[dict[str, Any]] = []
    durations: list[int] = []
    for index, raw in enumerate(rows_raw):
        row = _strict_mapping(
            raw,
            _PUSHED_RESULT_LATENCY_ROW_FIELDS,
            code="pushed_result_pause_artifact_invalid",
        )
        try:
            call = execution_calls[index]
            expected_ordinal = getattr(call, "execution_ordinal")
            expected_request_sha256 = getattr(call, "request")["request_sha256"]
            expected_request_bytes = getattr(call, "request_byte_count")
        except Exception:
            _reject("pushed_result_pause_artifact_invalid")
        duration = row["duration_ns"]
        if (
            row["execution_ordinal"] != expected_ordinal
            or row["request_sha256"] != expected_request_sha256
            or row["request_byte_count"] != expected_request_bytes
            or not _is_sha256(row["request_sha256"])
            or type(row["execution_ordinal"]) is not int
            or row["execution_ordinal"] != index + 1
            or type(row["request_byte_count"]) is not int
            or row["request_byte_count"] <= 0
            or type(duration) is not int
            or duration <= 0
        ):
            _reject("pushed_result_pause_artifact_invalid")
        rows.append(row)
        durations.append(duration)
    projected = contract.projected_pilot_ns(durations)
    if (
        item["schema_version"] != PUSHED_RESULT_LATENCY_SCHEMA_VERSION
        or item["selection"]
        != "five_largest_request_bytes_desc_accession_asc"
        or item["remaining_order"]
        != "availability_acceptance_accession_ascending"
        or item["pilot_order_sha256"] != contract.canonical_sha256(rows)
        or item["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or item["remaining_count"] != contract.DEVELOPMENT_REMAINING_COUNT
        or item["formula"]
        != "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
        or item["projected_ns"] != projected
        or item["threshold_ns"] != contract.PILOT_PROJECTED_THRESHOLD_NS
        or item["pause_required"] is not True
        or projected <= contract.PILOT_PROJECTED_THRESHOLD_NS
        or item["timing_interval"]
        != (
            "monotonic_ns_after_durable_intent_before_transport_through_"
            "bounded_body_framing_and_close_before_persistence_or_semantic_parse"
        )
    ):
        _reject("pushed_result_pause_artifact_invalid")
    return item


def _pilot_pause_effect_report() -> dict[str, Any]:
    attempted = {
        "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
        "ollama_identity_http_requests": 4,
        "ollama_chat_generations": contract.DEVELOPMENT_PILOT_COUNT,
    }
    request_count = sum(attempted.values())
    return {
        "attempted_external_requests": attempted,
        "completed_effect_counts": contract.build_effect_budgets()["pilot_pause"],
        "request_intent_count": request_count,
        "response_count": request_count,
        "checkpoint_count": request_count + 1,
        "no_retry_repair_pull_fallback_paid_or_trading_effect": True,
    }


def _authenticate_result_continuation_prerequisites(
    *,
    parent: Mapping[str, Any],
    dependencies: PushedResultDependencies,
    store: Any,
    authority: Mapping[str, Any],
    plan: Any,
) -> str:
    """Replay exact private pilot -> public S -> public C at a result commit."""

    try:
        evidence = dependencies.rebuild_pilot_evidence(store=store, plan=plan)
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if type(evidence) is not tuple or len(evidence) != 2:
        _reject("pushed_result_pause_artifact_invalid")
    pilot_guard = _validate_replayed_pilot_guard(evidence[0])
    latency = _validate_replayed_pilot_latency(evidence[1], plan=plan)
    try:
        plan_manifest = _plain_json_mapping(
            getattr(plan, "manifest"),
            code="pushed_result_pause_artifact_invalid",
        )
        plan_manifest = contract.validate_self_sha256(
            plan_manifest, field="model_plan_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if (
        not _is_sha256(plan_manifest.get("model_plan_sha256"))
        or plan_manifest.get("remaining_order_sha256")
        != authority["request_order"]["remaining_order_sha256"]
    ):
        _reject("pushed_result_pause_artifact_invalid")
    pilot_effect = _pilot_pause_effect_report()
    yahoo_body_bytes = getattr(store.snapshot, "yahoo_body_bytes", None)
    if type(yahoo_body_bytes) is not int or yahoo_body_bytes <= 0:
        _reject("pushed_result_pause_artifact_invalid")
    try:
        expected_pause_raw = dependencies.build_public_pause_artifact(
            authority=authority,
            plan=plan,
            pilot_guard=pilot_guard,
            latency_receipt=latency,
            effect_report={
                **pilot_effect,
                "yahoo_body_bytes": yahoo_body_bytes,
            },
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    expected_pause = _strict_mapping(
        expected_pause_raw,
        _PUSHED_RESULT_PAUSE_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        expected_pause = contract.validate_self_sha256(
            expected_pause, field="pause_artifact_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    durations = [row["duration_ns"] for row in latency["pilot_rows"]]
    if (
        expected_pause["schema_version"] != PUSHED_RESULT_PAUSE_SCHEMA_VERSION
        or expected_pause["status"] != "paused_for_justification"
        or expected_pause["stage"] != contract.DEVELOPMENT_COMMAND
        or expected_pause["attempt_id"] != contract.DEVELOPMENT_ATTEMPT_ID
        or expected_pause["branch"] != contract.BRANCH_NAME
        or expected_pause["preflight_commit"] != parent["preflight_commit"]
        or expected_pause["implementation_commit"]
        != authority["implementation"]["commit"]
        or expected_pause["source_bridge_sha256"]
        != authority["source"]["bridge_sha256"]
        or expected_pause["science_projection_sha256"]
        != contract.SCIENCE_PROJECTION_SHA256
        or expected_pause["model_plan_sha256"]
        != plan_manifest["model_plan_sha256"]
        or expected_pause["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or expected_pause["pilot_durations_ns"] != durations
        or expected_pause["formula"] != latency["formula"]
        or expected_pause["projected_ns"] != latency["projected_ns"]
        or expected_pause["threshold_ns"] != latency["threshold_ns"]
        or expected_pause["strictly_greater_pause"] is not True
        or expected_pause["pilot_guard_sha256"]
        != pilot_guard["segment_guard_sha256"]
        or expected_pause["latency_receipt_sha256"]
        != latency["latency_receipt_sha256"]
        or expected_pause["remaining_order_sha256"]
        != plan_manifest["remaining_order_sha256"]
        or expected_pause["effect_report"] != pilot_effect
        or expected_pause["model_responses_opened"] is not False
        or expected_pause["market_values_opened"] is not False
        or expected_pause["sixth_generation_attempted"] is not False
        or expected_pause["continuation_requires_fresh_explicit_permission"]
        is not True
        or expected_pause["confirmation_and_final_opened"] is not False
        or expected_pause["privacy_passed"] is not True
    ):
        _reject("pushed_result_pause_artifact_invalid")
    expected_pause_bytes = contract.canonical_json_bytes(expected_pause)
    committed_pause = _strict_mapping(
        parent.get("pause_value"),
        _PUSHED_RESULT_PAUSE_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        committed_pause = contract.validate_self_sha256(
            committed_pause, field="pause_artifact_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if (
        committed_pause != expected_pause
        or parent.get("pause_blob", {}).get("payload") != expected_pause_bytes
    ):
        _reject("pushed_result_pause_artifact_invalid")
    try:
        expected_continuation = dependencies.build_continuation_preregistration(
            expected_pause
        )
    except Exception:
        _reject("pushed_result_continuation_invalid")
    if (
        type(expected_continuation) is not bytes
        or not expected_continuation
        or len(expected_continuation) > MAX_COMPARISON_BYTES
        or parent.get("continuation_blob", {}).get("payload")
        != expected_continuation
    ):
        _reject("pushed_result_continuation_invalid")
    return hashlib.sha256(expected_continuation).hexdigest()


def _classify_result_parent(root: Path, parent: str) -> dict[str, Any]:
    """Reconstruct F or F->S->C strictly from committed Git objects."""

    parent_parent = _git_single_parent(
        root, parent, code="pushed_result_parent_invalid"
    )
    direct = _changed_paths(root, parent_parent, parent)
    if direct == {contract.PREFLIGHT_ARTIFACT_PATH: "A"}:
        return {
            "authorized_parent": parent,
            "authorized_parent_kind": "preflight",
            "preflight_commit": parent,
            "pause_commit": None,
            "continuation_commit": None,
            "continuation_document_sha256": None,
            "pause_blob": None,
            "pause_value": None,
            "continuation_blob": None,
        }
    if direct != {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}:
        _reject("pushed_result_parent_invalid")
    continuation_commit = parent
    pause_commit = parent_parent
    preflight_commit = _git_single_parent(
        root, pause_commit, code="pushed_result_parent_invalid"
    )
    if (
        _changed_paths(root, preflight_commit, pause_commit)
        != {contract.PAUSE_ARTIFACT_PATH: "A"}
        or _changed_paths(root, preflight_commit, continuation_commit)
        != {
            contract.PAUSE_ARTIFACT_PATH: "A",
            contract.CONTINUATION_PREREGISTRATION_PATH: "A",
        }
    ):
        _reject("pushed_result_parent_invalid")
    pause_blob = _git_blob_material(
        root,
        pause_commit,
        contract.PAUSE_ARTIFACT_PATH,
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_pause_artifact_invalid",
    )
    pause_value = _parse_canonical_mapping_bytes(
        pause_blob["payload"],
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_pause_artifact_invalid",
    )
    continuation_blob = _git_blob_material(
        root,
        continuation_commit,
        contract.CONTINUATION_PREREGISTRATION_PATH,
        maximum=MAX_COMPARISON_BYTES,
        code="pushed_result_continuation_invalid",
    )
    return {
        "authorized_parent": continuation_commit,
        "authorized_parent_kind": "continuation",
        "preflight_commit": preflight_commit,
        "pause_commit": pause_commit,
        "continuation_commit": continuation_commit,
        "continuation_document_sha256": None,
        "pause_blob": pause_blob,
        "pause_value": pause_value,
        "continuation_blob": continuation_blob,
    }


def _validate_public_private_qualification_binding(
    public_artifact: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
) -> None:
    try:
        public_qualification = public_artifact["qualification"]
        private_qualification = private_manifest["qualification"]
    except (KeyError, TypeError):
        _reject("execution_preflight_public_private_mismatch")
    if public_qualification != private_qualification:
        _reject("execution_preflight_public_private_mismatch")


def _validate_public_private_commitment_binding(
    public_artifact: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
) -> None:
    """Bind every pushed aggregate to the private request authority."""

    try:
        bridge = validate_bridge_manifest(private_manifest["bridge"])
        commitments = validate_request_commitments(
            private_manifest["request_commitments"]
        )
        counts = public_artifact["counts"]
        aggregates = public_artifact["aggregates"]
    except (KeyError, TypeError):
        _reject("execution_preflight_public_private_mismatch")
    _validate_bridge_commitment_parity(bridge, commitments)
    expected_counts = {
        "documents": commitments["document_count"],
        "records": commitments["record_count"],
        "events": commitments["event_count"],
        "canonical_requests": commitments["request_count"],
        "pilots": commitments["pilot_count"],
        "remaining": commitments["remaining_count"],
        "filename_present": commitments["filename_present_count"],
        "filename_missing": commitments["filename_missing_count"],
        "minimum_request_byte_count": commitments[
            "minimum_request_byte_count"
        ],
        "maximum_request_byte_count": commitments[
            "maximum_request_byte_count"
        ],
    }
    expected_aggregates = {
        "bridge_sha256": bridge["bridge_sha256"],
        "legacy_projection_sha256": bridge["legacy_projection_sha256"],
        "compatibility_manifest_sha256": commitments[
            "compatibility_manifest_sha256"
        ],
        "universe_sha256": commitments["universe_sha256"],
        "content_manifest_sha256": commitments["content_manifest_sha256"],
        "calendar_sessions_sha256": commitments["calendar_sessions_sha256"],
        "universe_event_proofs_sha256": commitments[
            "universe_event_proofs_sha256"
        ],
        "documents_sha256": commitments["documents_sha256"],
        "records_sha256": commitments["records_sha256"],
        "events_sha256": commitments["events_sha256"],
        "preprocessed_events_sha256": commitments[
            "preprocessed_events_sha256"
        ],
        "canonical_requests_sha256": commitments[
            "canonical_requests_sha256"
        ],
        "model_slice_sha256": commitments["model_slice_sha256"],
        "canonical_request_index_sha256": commitments[
            "canonical_request_index_sha256"
        ],
        "model_plan_sha256": commitments["model_plan_sha256"],
        "pilot_order_sha256": commitments["pilot_order_sha256"],
        "remaining_order_sha256": commitments["remaining_order_sha256"],
        "source_order_sha256": commitments["source_order_sha256"],
        "prior_links_sha256": commitments["prior_links_sha256"],
    }
    if (
        any(counts.get(key) != value for key, value in expected_counts.items())
        or any(
            aggregates.get(key) != value
            for key, value in expected_aggregates.items()
        )
    ):
        _reject("execution_preflight_public_private_mismatch")


def _build_execution_authority_receipt(
    *,
    public_artifact: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    preflight_commit: str,
    preflight_tree: str,
    implementation_commit: str,
    implementation_tree: str,
    public_artifact_git_blob_sha1: str,
    public_artifact_literal_sha256: str,
    direct_head: bool,
) -> dict[str, Any]:
    """Pure production seam from authenticated preflight values to F receipt."""

    public = validate_public_preflight_artifact(public_artifact)
    private = validate_private_preflight_manifest(private_manifest)
    _validate_public_private_qualification_binding(public, private)
    _validate_public_private_commitment_binding(public, private)
    if (
        type(direct_head) is not bool
        or any(
            not _is_sha1(value)
            for value in (
                preflight_commit,
                preflight_tree,
                implementation_commit,
                implementation_tree,
                public_artifact_git_blob_sha1,
            )
        )
        or not _is_sha256(public_artifact_literal_sha256)
        or hashlib.sha256(_artifact_bytes(public)).hexdigest()
        != public_artifact_literal_sha256
        or public["implementation_commit"] != implementation_commit
        or public["implementation_tree"] != implementation_tree
        or public["private_manifest_sha256"]
        != private["private_manifest_sha256"]
        or public["private_manifest_literal_sha256"]
        != hashlib.sha256(_artifact_bytes(private)).hexdigest()
        or public["runtime_binding_authority_sha256"]
        != private["runtime_binding_authority_sha256"]
    ):
        _reject("execution_preflight_public_private_mismatch")
    aggregates = public["aggregates"]
    counts = public["counts"]
    unsigned = {
        "schema_version": EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": (
            "direct_pushed_preflight"
            if direct_head
            else "descendant_reconstruction"
        ),
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": preflight_commit,
        "preflight_tree": preflight_tree,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "public_artifact_git_blob_sha1": public_artifact_git_blob_sha1,
        "public_artifact_sha256": public["public_artifact_sha256"],
        "public_artifact_literal_sha256": public_artifact_literal_sha256,
        "private_manifest_sha256": private["private_manifest_sha256"],
        "private_manifest_literal_sha256": public[
            "private_manifest_literal_sha256"
        ],
        "bridge_sha256": aggregates["bridge_sha256"],
        "compatibility_manifest_sha256": aggregates[
            "compatibility_manifest_sha256"
        ],
        "universe_sha256": aggregates["universe_sha256"],
        "content_manifest_sha256": aggregates["content_manifest_sha256"],
        "calendar_sessions_sha256": aggregates["calendar_sessions_sha256"],
        "universe_event_proofs_sha256": aggregates[
            "universe_event_proofs_sha256"
        ],
        "source_order_sha256": aggregates["source_order_sha256"],
        "prior_links_sha256": aggregates["prior_links_sha256"],
        "canonical_requests_sha256": aggregates[
            "canonical_requests_sha256"
        ],
        "model_slice_sha256": aggregates["model_slice_sha256"],
        "canonical_request_index_sha256": aggregates[
            "canonical_request_index_sha256"
        ],
        "model_plan_sha256": aggregates["model_plan_sha256"],
        "pilot_order_sha256": aggregates["pilot_order_sha256"],
        "remaining_order_sha256": aggregates["remaining_order_sha256"],
        "request_count": counts["canonical_requests"],
        "pilot_count": counts["pilots"],
        "remaining_count": counts["remaining"],
        "minimum_request_byte_count": counts["minimum_request_byte_count"],
        "maximum_request_byte_count": counts["maximum_request_byte_count"],
        "production_source_inventory_sha256": aggregates[
            "production_source_inventory_sha256"
        ],
        "test_source_inventory_sha256": aggregates[
            "test_source_inventory_sha256"
        ],
        "runtime_binding_authority_sha256": public[
            "runtime_binding_authority_sha256"
        ],
        "effect_counts": copy.deepcopy(public["effect_counts"]),
        "pushed_preflight_gate_passed": True,
        "development_authorized": direct_head,
    }
    return _self_hash(unsigned, "execution_authority_sha256")


def _authenticate_preflight_revision(
    root: Path,
    *,
    preflight_commit: str,
    direct_head: bool,
    allow_publication_dirty: bool = False,
) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    current_head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tree = _git(root, "rev-parse", f"{preflight_commit}^{{tree}}")
    parent = _git(root, "rev-parse", f"{preflight_commit}^")
    parent_tree = _git(root, "rev-parse", f"{parent}^{{tree}}")
    if not all(
        type(item) is str
        for item in (
            branch,
            current_head,
            remote,
            status,
            tree,
            parent,
            parent_tree,
        )
    ):
        _reject("execution_preflight_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or status != "" and not allow_publication_dirty
        or (direct_head and (current_head != preflight_commit or remote != preflight_commit))
    ):
        _reject("execution_preflight_ancestry_invalid")
    changed = _changed_paths(root, parent, preflight_commit)

    private_manifest, private_payload, completion, _completion_payload = (
        _read_execution_private_state(root)
    )
    repository = validate_repository_snapshot(private_manifest["repository"])
    if (
        repository["ancestry"]["commit"] != parent
        or repository["ancestry"]["tree"] != parent_tree
    ):
        _reject("execution_preflight_implementation_invalid")
    preflight_ancestry = {
        "branch": branch,
        "commit": preflight_commit,
        "tree": tree,
        "parent": parent,
        "implementation_commit": parent,
        "implementation_tree": parent_tree,
        # In descendant reconstruction these are the authenticated historical F
        # identity, not a claim about current HEAD.  The caller separately proves
        # the exact S/C descendant chain and current remote equality.
        "local_head": preflight_commit,
        "remote_head": preflight_commit,
        "changed_paths": changed,
        # A recovery caller proves the only allowed dirty paths separately;
        # this Boolean authenticates the historical F commit itself.
        "clean_worktree": status == "" or allow_publication_dirty,
        "implementation_authenticated": True,
        "private_replay_passed": True,
        "zero_effects_verified": True,
        "preflight_run_count": 1,
    }
    try:
        contract.validate_preflight_ancestry(preflight_ancestry)
    except contract.ContractViolation:
        _reject("execution_preflight_ancestry_invalid")

    public_path = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    try:
        public_payload = _read_regular_file(
            public_path,
            maximum=1024 * 1024,
            code="execution_preflight_public_artifact_invalid",
        )
    except OSError:
        _reject("execution_preflight_public_artifact_invalid")
    public_artifact = validate_public_preflight_artifact(
        _parse_artifact_bytes(
            public_payload,
            maximum=1024 * 1024,
            code="execution_preflight_public_artifact_invalid",
        )
    )
    public_literal = hashlib.sha256(public_payload).hexdigest()
    public_blob = _git(
        root,
        "rev-parse",
        f"{preflight_commit}:{contract.PREFLIGHT_ARTIFACT_PATH}",
    )
    public_git_payload = _git(
        root,
        "show",
        f"{preflight_commit}:{contract.PREFLIGHT_ARTIFACT_PATH}",
        binary=True,
    )
    _validate_public_private_qualification_binding(
        public_artifact, private_manifest
    )
    _validate_public_private_commitment_binding(
        public_artifact, private_manifest
    )
    if (
        not _is_sha1(public_blob)
        or type(public_git_payload) is not bytes
        or public_git_payload != public_payload
        or completion["public_artifact_sha256"]
        != public_artifact["public_artifact_sha256"]
        or completion["public_artifact_literal_sha256"] != public_literal
        or public_artifact["private_manifest_sha256"]
        != private_manifest["private_manifest_sha256"]
        or public_artifact["private_manifest_literal_sha256"]
        != hashlib.sha256(private_payload).hexdigest()
        or public_artifact["runtime_binding_authority_sha256"]
        != private_manifest["runtime_binding_authority_sha256"]
        or public_artifact["implementation_commit"] != parent
        or public_artifact["implementation_tree"] != parent_tree
    ):
        _reject("execution_preflight_public_private_mismatch")

    for row in repository["source_inventory"]:
        working_path = root / Path(row["path"])
        try:
            working = _read_regular_file(
                working_path,
                maximum=None,
                code="execution_preflight_source_inventory_invalid",
            )
        except OSError:
            _reject("execution_preflight_source_inventory_invalid")
        blob = _git(root, "show", f"{preflight_commit}:{row['path']}", binary=True)
        blob_sha1 = _git(root, "rev-parse", f"{preflight_commit}:{row['path']}")
        if (
            type(blob) is not bytes
            or type(blob_sha1) is not str
            or working != blob
            or hashlib.sha256(blob).hexdigest() != row["literal_sha256"]
            or blob_sha1 != row["git_blob_sha1"]
            or len(blob) != row["byte_count"]
        ):
            _reject("execution_preflight_source_inventory_invalid")

    contact = load_private_contact(root)
    forbidden = (
        contact.encode("utf-8"),
        str(root).encode("utf-8"),
        str((root / V38_PRIVATE_ROOT).resolve(strict=False)).encode("utf-8"),
        str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)).encode(
            "utf-8"
        ),
    )
    _scan_private_bytes(private_payload, forbidden)
    _scan_qualification_private_tree(
        root,
        forbidden=_contact_v38_private_tokens(root, contact),
        code="execution_preflight_private_privacy_failed",
    )
    _scan_public_bytes(public_payload, forbidden)

    return _build_execution_authority_receipt(
        public_artifact=public_artifact,
        private_manifest=private_manifest,
        preflight_commit=preflight_commit,
        preflight_tree=tree,
        implementation_commit=parent,
        implementation_tree=parent_tree,
        public_artifact_git_blob_sha1=public_blob,
        public_artifact_literal_sha256=public_literal,
        direct_head=direct_head,
    )


def authenticate_execution_preflight(repo_root: Path) -> dict[str, Any]:
    """Authenticate pushed preflight F at exact local/remote HEAD, read-only."""

    root = _canonical_repo_root(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    if type(commit) is not str or type(parent) is not str:
        _reject("execution_preflight_git_invalid")
    if _changed_paths(root, parent, commit) != {
        contract.PREFLIGHT_ARTIFACT_PATH: "A"
    }:
        # Direct authentication is intentionally valid only while F itself is
        # the pushed HEAD.  A pause or continuation descendant must use the
        # reconstruction path, which also validates its complete ancestry.
        _reject("execution_preflight_ancestry_invalid")
    receipt = _authenticate_preflight_revision(
        root,
        preflight_commit=commit,
        direct_head=True,
    )
    _initialize_runtime_consumption_session(
        root, allowed_invocations={"development"}
    )
    return receipt


def _store_execution_authority(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact stable nine-key authority bound by AttemptStore."""

    return {
        "plan": contract.CONTRACT_MANIFEST_SHA256,
        "attempt": contract.DEVELOPMENT_ATTEMPT_ID,
        "implementation": {
            "commit": receipt["implementation_commit"],
            "tree": receipt["implementation_tree"],
            "production_source_inventory_sha256": receipt[
                "production_source_inventory_sha256"
            ],
            "test_source_inventory_sha256": receipt[
                "test_source_inventory_sha256"
            ],
        },
        "preflight": {
            "commit": receipt["preflight_commit"],
            "tree": receipt["preflight_tree"],
            "public_artifact_sha256": receipt["public_artifact_sha256"],
            "public_artifact_literal_sha256": receipt[
                "public_artifact_literal_sha256"
            ],
            "private_manifest_sha256": receipt["private_manifest_sha256"],
            "private_manifest_literal_sha256": receipt[
                "private_manifest_literal_sha256"
            ],
            "runtime_binding_authority_sha256": receipt[
                "runtime_binding_authority_sha256"
            ],
        },
        "source": {
            "base_commit": contract.V38_SOURCE_COMMIT,
            "base_tree": contract.V38_SOURCE_TREE,
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
            "bridge_sha256": receipt["bridge_sha256"],
            "compatibility_manifest_sha256": receipt[
                "compatibility_manifest_sha256"
            ],
            "universe_sha256": receipt["universe_sha256"],
            "content_manifest_sha256": receipt["content_manifest_sha256"],
            "calendar_sessions_sha256": receipt["calendar_sessions_sha256"],
            "universe_event_proofs_sha256": receipt[
                "universe_event_proofs_sha256"
            ],
            "source_order_sha256": receipt["source_order_sha256"],
            "prior_links_sha256": receipt["prior_links_sha256"],
        },
        "science": {"projection_sha256": contract.SCIENCE_PROJECTION_SHA256},
        "effect_budget": contract.build_effect_budgets(),
        "request_order": {
            "count": receipt["request_count"],
            "remaining_count": receipt["remaining_count"],
            "canonical_requests_sha256": receipt["canonical_requests_sha256"],
            "model_slice_sha256": receipt["model_slice_sha256"],
            "canonical_request_index_sha256": receipt[
                "canonical_request_index_sha256"
            ],
            "model_plan_sha256": receipt["model_plan_sha256"],
            "remaining_order_sha256": receipt["remaining_order_sha256"],
            "minimum_request_byte_count": receipt[
                "minimum_request_byte_count"
            ],
            "maximum_request_byte_count": receipt[
                "maximum_request_byte_count"
            ],
        },
        "pilot_order": {
            "count": receipt["pilot_count"],
            "pilot_order_sha256": receipt["pilot_order_sha256"],
        },
    }


def reconstruct_frozen_execution_authority(repo_root: Path) -> dict[str, Any]:
    """Rebuild exact F authority under an allowed S or C descendant.

    This is read-only and does not validate fresh user permission or authorize a
    continuation effect.  It exists so status/replay can reopen the original
    AttemptStore authority after the preregistered pause commits.
    """

    root = _canonical_repo_root(repo_root)
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if not all(type(item) is str for item in (branch, head, remote, status)):
        _reject("execution_preflight_descendant_invalid")
    if (
        branch != contract.BRANCH_NAME
        or head != remote
        or status != ""
    ):
        _reject("execution_preflight_descendant_invalid")
    parent = _git_single_parent(
        root, head, code="execution_preflight_descendant_invalid"
    )

    direct_delta = _changed_paths(root, parent, head)
    if direct_delta == {contract.PAUSE_ARTIFACT_PATH: "A"}:
        preflight_commit = parent
    elif direct_delta == {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}:
        pause_commit = parent
        preflight_commit = _git_single_parent(
            root, pause_commit, code="execution_preflight_descendant_invalid"
        )
        if _changed_paths(root, preflight_commit, pause_commit) != {
            contract.PAUSE_ARTIFACT_PATH: "A"
        }:
            _reject("execution_preflight_descendant_invalid")
        if _changed_paths(root, preflight_commit, head) != {
            contract.PAUSE_ARTIFACT_PATH: "A",
            contract.CONTINUATION_PREREGISTRATION_PATH: "A",
        }:
            _reject("execution_preflight_descendant_invalid")
    else:
        _reject("execution_preflight_descendant_invalid")

    receipt = _authenticate_preflight_revision(
        root,
        preflight_commit=preflight_commit,
        direct_head=False,
    )
    if receipt["development_authorized"] is not False:
        _reject("execution_preflight_descendant_invalid")
    _initialize_runtime_consumption_session(
        root, allowed_invocations={"continuation"}
    )
    return _store_execution_authority(receipt)


def load_execution_authority(repo_root: Path) -> dict[str, Any]:
    """Load stable AttemptStore authority at F, S, C, or clean pushed R.

    Reconstructing authority at R is only for safe status/replay access.  It
    does not claim that the separate pushed-result gate passed and cannot
    authorize another development effect.
    """

    root = _canonical_repo_root(repo_root)
    head = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    if type(head) is not str or type(parent) is not str:
        _reject("execution_preflight_git_invalid")
    delta = _changed_paths(root, parent, head)
    if delta == {contract.PREFLIGHT_ARTIFACT_PATH: "A"}:
        return _store_execution_authority(authenticate_execution_preflight(root))
    if delta == {
        contract.RESULT_ARTIFACT_PATH: "A",
        contract.COMPARISON_PATH: "M",
    }:
        result_state = _inspect_result_git(root)
        result_parent = _classify_result_parent(root, result_state["parent"])
        receipt = _authenticate_preflight_revision(
            root,
            preflight_commit=result_parent["preflight_commit"],
            direct_head=False,
        )
        return _store_execution_authority(receipt)
    return reconstruct_frozen_execution_authority(root)


def _publication_path_state(root: Path) -> dict[str, Any]:
    """Inspect only the three idempotent partial-publication paths."""

    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(
        root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
    )
    staged = _git(root, "diff", "--cached", "--name-status")
    tracked = _git(root, "diff", "--name-status")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    if not all(
        type(item) is str
        for item in (branch, head, remote, staged, tracked, untracked)
    ):
        _reject("publication_recovery_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or not _is_sha1(head)
        or head != remote
        or staged != ""
    ):
        _reject("publication_recovery_git_invalid")
    tracked_rows = tracked.splitlines() if tracked else []
    allowed_tracked = {f"M\t{contract.COMPARISON_PATH}"}
    if any(row not in allowed_tracked for row in tracked_rows) or len(tracked_rows) > 1:
        _reject("publication_recovery_dirty_invalid")
    pause_pending_path = f"{contract.PAUSE_ARTIFACT_PATH}.v315-pending"
    result_pending_path = f"{contract.RESULT_ARTIFACT_PATH}.v315-pending"
    comparison_pending_path = f"{contract.COMPARISON_PATH}.v315-pending"
    untracked_paths = set(untracked.splitlines()) if untracked else set()
    if len(untracked_paths) != len(untracked.splitlines()) or not untracked_paths <= {
        contract.PAUSE_ARTIFACT_PATH,
        pause_pending_path,
        contract.RESULT_ARTIFACT_PATH,
        result_pending_path,
        comparison_pending_path,
    }:
        _reject("publication_recovery_dirty_invalid")

    files: dict[str, dict[str, Any] | None] = {}
    for relative in (
        contract.RESULT_ARTIFACT_PATH,
        result_pending_path,
        contract.PAUSE_ARTIFACT_PATH,
        pause_pending_path,
        contract.COMPARISON_PATH,
        comparison_pending_path,
    ):
        path = root / Path(relative)
        try:
            details = path.lstat()
        except FileNotFoundError:
            files[relative] = None
            continue
        except OSError:
            _reject("publication_recovery_dirty_invalid")
        try:
            payload = _read_regular_file(
                path,
                maximum=None,
                code="publication_recovery_dirty_invalid",
            )
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            _reject("publication_recovery_dirty_invalid")
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            or details.st_nlink not in {0, 1}
            or details.st_size != len(payload)
        ):
            _reject("publication_recovery_dirty_invalid")
        files[relative] = {
            "literal_sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "payload": payload,
        }
    if (
        files[contract.COMPARISON_PATH] is None
        or any(
            (files[path] is not None) != (path in untracked_paths)
            for path in (
                contract.RESULT_ARTIFACT_PATH,
                result_pending_path,
                pause_pending_path,
                comparison_pending_path,
            )
        )
    ):
        _reject("publication_recovery_dirty_invalid")
    return {
        "branch": branch,
        "head": head,
        "remote": remote,
        "tracked_rows": tracked_rows,
        "untracked_paths": sorted(untracked_paths),
        "files": files,
    }


def build_production_publication_recovery_dependencies(
) -> PublicationRecoveryDependencies:
    """Bind terminal-store replay for publication recovery without effects."""

    try:
        from .sec_gemma_lean_science_v315_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v315_runner import (
            _pilot_evidence,
            build_attempt_authority,
            build_comparison_update,
            build_effect_report,
            build_model_plan,
            build_preflight_request_commitments,
            build_public_pause_artifact,
            build_public_terminal_artifact,
        )
        from .sec_gemma_lean_science_v315_store import AttemptStore
    except Exception:
        _reject("publication_recovery_dependency_unavailable")

    def rebuild_pause(
        *,
        root: Path,
        store: Any,
        authority: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        contact = load_private_contact(root)
        source = authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact,
        )
        projection = build_streaming_science_projection(source)
        commitments = build_preflight_request_commitments(projection)
        plan = build_model_plan(projection)
        rebuilt = build_attempt_authority(
            execution_authority=authority,
            projection=projection,
            commitments=commitments,
            plan=plan,
        )
        if _plain_json_mapping(
            rebuilt, code="publication_recovery_authority_invalid"
        ) != _plain_json_mapping(
            authority, code="publication_recovery_authority_invalid"
        ):
            _reject("publication_recovery_authority_invalid")
        guard, latency = _pilot_evidence(store, plan=plan)
        effects = build_effect_report(store.snapshot)
        try:
            contract.validate_effect_counts(
                effects["completed_effect_counts"], route="pilot_pause"
            )
        except Exception:
            _reject("publication_recovery_pause_invalid")
        return build_public_pause_artifact(
            authority=authority,
            plan=plan,
            pilot_guard=guard,
            latency_receipt=latency,
            effect_report=effects,
        )

    return PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda root, commit: (
            _authenticate_preflight_revision(
                root,
                preflight_commit=commit,
                direct_head=False,
                allow_publication_dirty=True,
            )
        ),
        open_store=lambda path, authority: AttemptStore.open(
            path, authority=authority
        ),
        build_public_terminal_artifact=build_public_terminal_artifact,
        build_comparison_update=build_comparison_update,
        rebuild_public_pause_artifact=rebuild_pause,
    )


def load_publication_recovery_authority(
    repo_root: Path,
    *,
    dependencies: PublicationRecoveryDependencies | None = None,
) -> dict[str, Any]:
    """Recover F/C authority only for an exact partial pause/result publication.

    This path can reopen an already terminal-sealed store and compare bytes. It
    returns the same stable nine-key authority used by the attempt; it grants
    no permission to perform or retry any external effect.
    """

    root = _canonical_repo_root(repo_root)
    deps = (
        dependencies
        if dependencies is not None
        else build_production_publication_recovery_dependencies()
    )
    production_runtime = dependencies is None
    if type(deps) is not PublicationRecoveryDependencies:
        _reject("publication_recovery_dependencies_invalid")
    store: Any | None = None
    try:
        before = _publication_path_state(root)
        parent = _classify_result_parent(root, before["head"])
        if parent["authorized_parent_kind"] not in {"preflight", "continuation"}:
            _reject("publication_recovery_parent_invalid")
        preflight_receipt = deps.authenticate_preflight_revision(
            root, parent["preflight_commit"]
        )
        if (
            not isinstance(preflight_receipt, Mapping)
            or preflight_receipt.get("preflight_commit")
            != parent["preflight_commit"]
            or preflight_receipt.get("pushed_preflight_gate_passed") is not True
            or preflight_receipt.get("development_authorized") is not False
        ):
            _reject("publication_recovery_authority_invalid")
        if production_runtime:
            _initialize_runtime_consumption_session(
                root, allowed_invocations={"publication_recovery"}
            )
        authority = _store_execution_authority(preflight_receipt)
        store = deps.open_store(
            root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE), authority
        )
        snapshot = store.snapshot
        try:
            receipt = store.committed_terminal_evidence()
        except Exception:
            receipt = None
        files = before["files"]
        pause_pending_path = f"{contract.PAUSE_ARTIFACT_PATH}.v315-pending"
        result_pending_path = f"{contract.RESULT_ARTIFACT_PATH}.v315-pending"
        comparison_pending_path = f"{contract.COMPARISON_PATH}.v315-pending"
        committed_comparison = _git_blob_material(
            root,
            before["head"],
            contract.COMPARISON_PATH,
            maximum=MAX_COMPARISON_BYTES,
            code="publication_recovery_comparison_invalid",
        )["payload"]
        if receipt is None:
            pause_builder = deps.rebuild_public_pause_artifact
            if (
                getattr(snapshot, "status", None) != "paused"
                or getattr(snapshot, "paused", None) is not True
                or getattr(snapshot, "continuation_authorized", None) is not False
                or parent["authorized_parent_kind"] != "preflight"
                or pause_builder is None
            ):
                _reject("publication_recovery_pause_invalid")
            expected_pause_raw = pause_builder(
                root=root, store=store, authority=authority
            )
            expected_pause = _plain_json_mapping(
                expected_pause_raw, code="publication_recovery_pause_invalid"
            )
            pause_bytes = contract.canonical_json_bytes(expected_pause)
            for path in (contract.PAUSE_ARTIFACT_PATH, pause_pending_path):
                if files[path] is not None and files[path]["payload"] != pause_bytes:
                    _reject("publication_recovery_pause_invalid")
            if (
                (
                    files[contract.PAUSE_ARTIFACT_PATH] is not None
                    and files[pause_pending_path] is not None
                )
                or
                files[contract.RESULT_ARTIFACT_PATH] is not None
                or files[result_pending_path] is not None
                or files[comparison_pending_path] is not None
                or files[contract.COMPARISON_PATH]["payload"]
                != committed_comparison
                or before["tracked_rows"]
                or (
                    files[contract.PAUSE_ARTIFACT_PATH] is not None
                    and contract.PAUSE_ARTIFACT_PATH
                    not in before["untracked_paths"]
                )
            ):
                _reject("publication_recovery_dirty_invalid")
        else:
            if (
                getattr(snapshot, "status", None)
                not in {"completed", "rejected", "indeterminate"}
                or getattr(snapshot, "journal_head_sha256", None)
                != getattr(receipt, "event_sha256", None)
            ):
                _reject("publication_recovery_terminal_invalid")
            if parent["authorized_parent_kind"] == "continuation":
                if (
                    getattr(snapshot, "continuation_authorized", None) is not True
                    or getattr(snapshot, "continuation_commit", None)
                    != parent["continuation_commit"]
                ):
                    _reject("publication_recovery_parent_invalid")
            elif getattr(snapshot, "continuation_commit", None) is not None:
                _reject("publication_recovery_parent_invalid")
            public = deps.build_public_terminal_artifact(
                authority=authority, terminal_receipt=receipt
            )
            public_value = _plain_json_mapping(
                public, code="publication_recovery_terminal_invalid"
            )
            if (
                public_value.get("invocation_parent") != before["head"]
                or public_value.get("invocation_parent_kind")
                != parent["authorized_parent_kind"]
            ):
                _reject("publication_recovery_parent_invalid")
            result_bytes = contract.canonical_json_bytes(public_value)
            derived_comparison = deps.build_comparison_update(
                committed_comparison, public_value
            )
            if type(derived_comparison) is not bytes:
                _reject("publication_recovery_comparison_invalid")
            for path in (contract.RESULT_ARTIFACT_PATH, result_pending_path):
                if files[path] is not None and files[path]["payload"] != result_bytes:
                    _reject("publication_recovery_result_invalid")
            comparison_file = files[contract.COMPARISON_PATH]
            comparison_pending = files[comparison_pending_path]
            if comparison_file["payload"] not in {
                committed_comparison,
                derived_comparison,
            }:
                _reject("publication_recovery_comparison_invalid")
            if (
                (
                    files[contract.RESULT_ARTIFACT_PATH] is not None
                    and files[result_pending_path] is not None
                )
                or (
                    comparison_pending is not None
                    and comparison_pending["payload"] != derived_comparison
                )
                or (
                    comparison_file["payload"] == derived_comparison
                    and comparison_pending is not None
                )
            ):
                _reject("publication_recovery_pending_invalid")
            comparison_is_dirty = comparison_file["payload"] == derived_comparison
            if (f"M\t{contract.COMPARISON_PATH}" in before["tracked_rows"]) != (
                comparison_is_dirty and derived_comparison != committed_comparison
            ):
                _reject("publication_recovery_dirty_invalid")
            pause_file = files[contract.PAUSE_ARTIFACT_PATH]
            if (
                files[pause_pending_path] is not None
                or parent["authorized_parent_kind"] == "preflight"
                and pause_file is not None
                or parent["authorized_parent_kind"] == "continuation"
                and (
                    pause_file is None
                    or contract.PAUSE_ARTIFACT_PATH in before["untracked_paths"]
                )
            ):
                _reject("publication_recovery_dirty_invalid")
        store.close()
        store = None
        if _publication_path_state(root) != before:
            _reject("publication_recovery_state_changed")
        return authority
    except V315PreflightError:
        raise
    except Exception:
        _reject("publication_recovery_dependency_failed")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


_COMPLETE_PRIVATE_TERMINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "invocation_parent",
        "invocation_parent_kind",
        "route",
        "authority_sha256",
        "bridge_manifest",
        "model_plan_manifest",
        "runtime_segment_guards",
        "runtime_aggregate",
        "latency_receipt",
        "stage_slice_sha256",
        "semantic_payload",
        "semantic_payload_sha256",
        "deterministic_payload",
        "deterministic_payload_sha256",
        "science_summary",
        "science_summary_sha256",
        "effect_report",
        "market_values_opened",
        "model_responses_opened",
        "confirmation_and_final_opened",
        "raw_sec_yahoo_or_gemma_response_copied",
        "private_terminal_material_sha256",
    }
)
_FAILURE_PRIVATE_TERMINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "invocation_parent",
        "invocation_parent_kind",
        "route",
        "authority_sha256",
        "terminal_status",
        "terminal_code",
        "effect_report",
        "journal_head_before_terminal_sha256",
        "closed_segment_generation_counts",
        "market_values_opened",
        "model_responses_opened",
        "source_authenticated",
        "confirmation_and_final_opened",
        "redacted_error_only",
        "private_terminal_material_sha256",
    }
)


def _validate_result_self_hash(value: Any, *, field: str, code: str) -> dict[str, Any]:
    try:
        plain = _plain_json_mapping(value, code=code)
        return contract.validate_self_sha256(plain, field=field)
    except Exception:
        _reject(code)


def _expected_result_route(status: str, parent_kind: str) -> str:
    if parent_kind == "continuation":
        return "paused_resumed"
    if parent_kind != "preflight":
        _reject("pushed_result_parent_invalid")
    return "indeterminate_before_pause" if status == "indeterminate" else "normal"


def _validate_public_result_basics(
    value: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _validate_result_self_hash(
        value,
        field="development_result_sha256",
        code="pushed_result_artifact_invalid",
    )
    terminal_status = item.get("terminal_store_status")
    outcome = item.get("status")
    parent_kind = parent["authorized_parent_kind"]
    if terminal_status not in {"completed", "rejected", "indeterminate"}:
        _reject("pushed_result_artifact_invalid")
    expected_route = _expected_result_route(terminal_status, parent_kind)
    if (
        item.get("stage") != contract.DEVELOPMENT_COMMAND
        or item.get("attempt_id") != contract.DEVELOPMENT_ATTEMPT_ID
        or item.get("branch") != contract.BRANCH_NAME
        or item.get("route") != expected_route
        or item.get("invocation_parent") != parent["authorized_parent"]
        or item.get("invocation_parent_kind") != parent_kind
        or item.get("science_projection_sha256")
        != contract.SCIENCE_PROJECTION_SHA256
        or item.get("contract_manifest_sha256")
        != contract.CONTRACT_MANIFEST_SHA256
        or item.get("real_money_authorized") is not False
        or type(item.get("market_values_opened")) is not bool
        or type(item.get("model_responses_opened")) is not bool
        or item.get("model_responses_opened") is True
        and item.get("market_values_opened") is not True
    ):
        _reject("pushed_result_artifact_invalid")
    terminal_code = item.get("terminal_code")
    if (
        type(terminal_code) is not str
        or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", terminal_code) is None
    ):
        _reject("pushed_result_artifact_invalid")
    if terminal_status == "completed":
        if outcome not in {"passed", "failed_gate"}:
            _reject("pushed_result_artifact_invalid")
    elif outcome != terminal_status:
        _reject("pushed_result_artifact_invalid")
    chronology = item.get("chronology")
    privacy = item.get("privacy")
    limitations = item.get("evidence_limitations")
    if (
        not isinstance(chronology, Mapping)
        or chronology.get("development_only") is not True
        or chronology.get("development_end") != contract.DEVELOPMENT_CORPUS_END
        or chronology.get("confirmation_2019_2023_opened") is not False
        or chronology.get("final_2024_plus_opened") is not False
        or not isinstance(privacy, Mapping)
        or privacy.get("privacy_passed") is not True
        or any(
            privacy.get(key) is not False
            for key in (
                "readable_sec_contact_published",
                "accessions_urls_filenames_offsets_or_bodies_published",
                "canonical_model_requests_published",
                "raw_sec_yahoo_or_gemma_responses_published",
                "realized_transport_metadata_published",
            )
        )
        or privacy.get("redacted_errors_only") is not True
        or type(limitations) is not list
        or "confirmation_and_final_remain_closed" not in limitations
        or "no_real_money_execution_authorized" not in limitations
    ):
        _reject("pushed_result_artifact_invalid")
    if preflight_receipt is not None and (
        item.get("preflight_commit") != preflight_receipt.get("preflight_commit")
        or item.get("preflight_tree") != preflight_receipt.get("preflight_tree")
        or item.get("implementation_commit")
        != preflight_receipt.get("implementation_commit")
        or item.get("implementation_tree")
        != preflight_receipt.get("implementation_tree")
    ):
        _reject("pushed_result_artifact_invalid")
    return item


def _validate_effect_report(
    value: Any,
    *,
    snapshot: Any,
    completed_terminal: bool,
    parent_kind: str,
) -> dict[str, Any]:
    expected_fields = {
        "attempted_external_requests",
        "completed_effect_counts",
        "request_intent_count",
        "response_count",
        "checkpoint_count",
        "yahoo_body_bytes",
        "no_retry_repair_pull_fallback_paid_or_trading_effect",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _reject("pushed_result_effects_invalid")
    item = _plain_json_mapping(value, code="pushed_result_effects_invalid")
    attempted = item["attempted_external_requests"]
    completed = item["completed_effect_counts"]
    expected_attempted = {
        "yahoo_requests",
        "ollama_identity_http_requests",
        "ollama_chat_generations",
    }
    expected_completed = set(contract.build_effect_budgets()["normal_complete"])
    if (
        not isinstance(attempted, Mapping)
        or set(attempted) != expected_attempted
        or not isinstance(completed, Mapping)
        or set(completed) != expected_completed
        or any(type(child) is not int or child < 0 for child in attempted.values())
        or any(type(child) is not int or child < 0 for child in completed.values())
        or item["no_retry_repair_pull_fallback_paid_or_trading_effect"] is not True
    ):
        _reject("pushed_result_effects_invalid")
    if (
        completed["sec_requests"] != 0
        or completed["experiment_family_sec_requests"]
        != contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
        or any(
            completed[key] != 0
            for key in (
                "retries",
                "repairs",
                "pulls",
                "fallbacks",
                "paid_calls",
                "confirmation_final_data_opens",
                "broker_effects",
                "real_money_effects",
            )
        )
    ):
        _reject("pushed_result_effects_invalid")
    snapshot_pairs = {
        "request_intent_count": "request_intent_count",
        "response_count": "response_count",
        "checkpoint_count": "checkpoint_count",
        "yahoo_body_bytes": "yahoo_body_bytes",
    }
    if any(
        type(item[field]) is not int
        or item[field] < 0
        or item[field] != getattr(snapshot, attribute, None)
        for field, attribute in snapshot_pairs.items()
    ):
        _reject("pushed_result_effects_invalid")
    observed_attempted = {
        "yahoo_requests": getattr(snapshot, "yahoo_intent_count", None),
        "ollama_identity_http_requests": getattr(
            snapshot, "identity_intent_count", None
        ),
        "ollama_chat_generations": getattr(snapshot, "gemma_intent_count", None),
    }
    observed_completed = {
        "yahoo_requests": getattr(snapshot, "yahoo_response_count", None),
        "ollama_identity_http_requests": getattr(
            snapshot, "identity_response_count", None
        ),
        "ollama_chat_generations": getattr(snapshot, "gemma_response_count", None),
    }
    if dict(attempted) != observed_attempted or any(
        completed[key] != observed_completed[key] for key in observed_completed
    ):
        _reject("pushed_result_effects_invalid")
    if (
        item["request_intent_count"] != sum(attempted.values())
        or item["response_count"]
        != sum(completed[key] for key in observed_completed)
        or item["checkpoint_count"] > item["response_count"] + 1
        or any(attempted[key] < completed[key] for key in attempted)
    ):
        _reject("pushed_result_effects_invalid")
    if completed_terminal:
        route = (
            "normal_complete"
            if parent_kind == "preflight"
            else "paused_resumed_complete"
        )
        try:
            contract.validate_effect_counts(completed, route=route)
        except Exception:
            _reject("pushed_result_effects_invalid")
        if any(attempted[key] != completed[key] for key in attempted):
            _reject("pushed_result_effects_invalid")
    else:
        maxima = {
            "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
            "ollama_identity_http_requests": 8 if parent_kind == "continuation" else 4,
            "ollama_chat_generations": contract.DEVELOPMENT_DOCUMENT_COUNT,
        }
        if any(
            attempted[key] > maxima[key] or completed[key] > maxima[key]
            for key in maxima
        ):
            _reject("pushed_result_effects_invalid")
    return item


def _validate_private_terminal_material(
    value: Any,
    *,
    receipt: Any,
    snapshot: Any,
    authority: Mapping[str, Any],
    projection_manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    effect_report: Mapping[str, Any],
) -> dict[str, Any]:
    status = getattr(receipt, "status", None)
    completed = status == "completed"
    expected_fields = (
        _COMPLETE_PRIVATE_TERMINAL_FIELDS
        if completed
        else _FAILURE_PRIVATE_TERMINAL_FIELDS
    )
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _reject("pushed_result_private_terminal_invalid")
    item = _validate_result_self_hash(
        value,
        field="private_terminal_material_sha256",
        code="pushed_result_private_terminal_invalid",
    )
    parent_kind = parent["authorized_parent_kind"]
    if (
        item.get("schema_version")
        != "aapl-sec-gemma-lean-science-v3-15-private-terminal-material-v1"
        or item.get("stage") != contract.DEVELOPMENT_COMMAND
        or item.get("attempt_id") != contract.DEVELOPMENT_ATTEMPT_ID
        or item.get("invocation_parent") != parent["authorized_parent"]
        or item.get("invocation_parent_kind") != parent_kind
        or item.get("route") != _expected_result_route(status, parent_kind)
        or item.get("authority_sha256") != contract.canonical_sha256(authority)
        or item.get("effect_report") != effect_report
        or item.get("confirmation_and_final_opened") is not False
        or type(item.get("market_values_opened")) is not bool
        or type(item.get("model_responses_opened")) is not bool
        or item.get("market_values_opened")
        != getattr(snapshot, "market_values_opened", None)
        or item.get("model_responses_opened")
        != getattr(snapshot, "model_responses_opened", None)
        or item.get("model_responses_opened") is True
        and item.get("market_values_opened") is not True
    ):
        _reject("pushed_result_private_terminal_invalid")
    if completed:
        semantic = item["semantic_payload"]
        deterministic = item["deterministic_payload"]
        summary = item["science_summary"]
        guards = item["runtime_segment_guards"]
        expected_guard_count = 1 if parent_kind == "preflight" else 2
        if (
            item.get("bridge_manifest") != projection_manifest
            or item.get("semantic_payload_sha256")
            != contract.canonical_sha256(semantic)
            or item.get("deterministic_payload_sha256")
            != contract.canonical_sha256(deterministic)
            or item.get("science_summary_sha256")
            != contract.canonical_sha256(summary)
            or type(guards) is not list
            or len(guards) != expected_guard_count
            or item.get("market_values_opened") is not True
            or item.get("model_responses_opened") is not True
            or item.get("raw_sec_yahoo_or_gemma_response_copied") is not False
            or not isinstance(summary, Mapping)
            or not isinstance(summary.get("gate_report"), Mapping)
            or type(summary["gate_report"].get("passed")) is not bool
            or summary.get("no_leverage_proofs_sha256")
            != contract.canonical_sha256(summary.get("no_leverage_proofs"))
        ):
            _reject("pushed_result_private_terminal_invalid")
    elif (
        item.get("terminal_status") != status
        or item.get("terminal_code") != getattr(receipt, "terminal_code", None)
        or item.get("journal_head_before_terminal_sha256")
        != getattr(receipt, "journal_head_before_terminal_sha256", None)
        or type(item.get("closed_segment_generation_counts")) is not list
        or any(
            type(child) is not int or child < 0
            for child in item["closed_segment_generation_counts"]
        )
        or type(item.get("source_authenticated")) is not bool
        or item.get("redacted_error_only") is not True
    ):
        _reject("pushed_result_private_terminal_invalid")
    if not completed:
        completed_counts = effect_report["completed_effect_counts"]
        full_identity = 4 if parent_kind == "preflight" else 8
        all_effects_complete = (
            completed_counts["yahoo_requests"] == contract.YAHOO_REQUEST_COUNT
            and completed_counts["ollama_identity_http_requests"] == full_identity
            and completed_counts["ollama_chat_generations"]
            == contract.DEVELOPMENT_DOCUMENT_COUNT
        )
        if (
            (item["market_values_opened"] or item["model_responses_opened"])
            and not all_effects_complete
            or item["source_authenticated"] is False
            and (
                item["market_values_opened"]
                or item["model_responses_opened"]
                or effect_report["request_intent_count"] != 0
                or effect_report["response_count"] != 0
            )
            or sum(item["closed_segment_generation_counts"])
            > completed_counts["ollama_chat_generations"]
        ):
            _reject("pushed_result_private_terminal_invalid")
    return item


def _comparison_inserted_bytes(before: bytes, after: bytes) -> bytes:
    prefix = 0
    limit = min(len(before), len(after))
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    before_left = len(before) - prefix
    after_left = len(after) - prefix
    while (
        suffix < before_left
        and suffix < after_left
        and before[-(suffix + 1)] == after[-(suffix + 1)]
    ):
        suffix += 1
    end = len(after) - suffix if suffix else len(after)
    return after[prefix:end]


def _validate_pushed_terminal_candidate_binding(
    *,
    result: Mapping[str, Any],
    snapshot: Any,
    receipt: Any,
    candidate: Any,
    committed_result_bytes: bytes,
    parent_comparison_bytes: bytes,
    committed_comparison_bytes: bytes,
) -> dict[str, Any]:
    """Bind the immutable store candidate to both committed public files."""

    code = "pushed_result_terminal_seal_invalid"
    receipt_evidence = _plain_json_mapping(
        getattr(receipt, "evidence", None), code=code
    )
    try:
        candidate_mapping = _plain_json_mapping(
            getattr(candidate, "candidate", None), code=code
        )
        _validate_self_hash(candidate_mapping, "candidate_sha256", code=code)
        candidate_bytes = contract.canonical_json_bytes(candidate_mapping)
        public_result_bytes = bytes(getattr(receipt, "public_result"))
        candidate_public_result = bytes(getattr(candidate, "public_result"))
        receipt_comparison_before = bytes(
            getattr(receipt, "comparison_before")
        )
        candidate_comparison_before = bytes(
            getattr(candidate, "comparison_before")
        )
        receipt_comparison_after = bytes(getattr(receipt, "comparison_after"))
        candidate_comparison_after = bytes(
            getattr(candidate, "comparison_after")
        )
    except Exception:
        _reject(code)
    if (
        type(committed_result_bytes) is not bytes
        or type(parent_comparison_bytes) is not bytes
        or type(committed_comparison_bytes) is not bytes
        or getattr(snapshot, "status", None)
        not in {"completed", "rejected", "indeterminate"}
        or getattr(snapshot, "status", None) != getattr(receipt, "status", None)
        or getattr(snapshot, "terminal_code", None)
        != getattr(receipt, "terminal_code", None)
        or getattr(snapshot, "journal_head_sha256", None)
        != getattr(receipt, "event_sha256", None)
        or result.get("terminal_store_status") != getattr(receipt, "status", None)
        or result.get("terminal_code") != getattr(receipt, "terminal_code", None)
        or result.get("terminal_material_sha256")
        != getattr(receipt, "terminal_material_sha256", None)
        or result.get("terminal_candidate_sequence") != 1
        or result.get("terminal_binding_required") is not True
        or getattr(receipt, "candidate_sha256", None)
        != getattr(candidate, "candidate_sha256", None)
        or getattr(receipt, "candidate_bytes", None)
        != getattr(candidate, "candidate_bytes", None)
        or getattr(receipt, "terminal_material_sha256", None)
        != getattr(candidate, "terminal_material_sha256", None)
        or getattr(receipt, "public_result_sha256", None)
        != getattr(candidate, "public_result_sha256", None)
        or getattr(receipt, "public_result_bytes", None)
        != getattr(candidate, "public_result_bytes", None)
        or getattr(receipt, "comparison_before_sha256", None)
        != getattr(candidate, "comparison_before_sha256", None)
        or getattr(receipt, "comparison_after_sha256", None)
        != getattr(candidate, "comparison_after_sha256", None)
        or getattr(receipt, "comparison_after_bytes", None)
        != getattr(candidate, "comparison_after_bytes", None)
        or candidate_mapping.get("candidate_sha256")
        != getattr(candidate, "candidate_sha256", None)
        or len(candidate_bytes) != getattr(candidate, "candidate_bytes", None)
        or contract.canonical_json_bytes(result) != public_result_bytes
        or public_result_bytes != candidate_public_result
        or public_result_bytes != committed_result_bytes
        or hashlib.sha256(public_result_bytes).hexdigest()
        != getattr(receipt, "public_result_sha256", None)
        or len(public_result_bytes) != getattr(receipt, "public_result_bytes", None)
        or receipt_comparison_before != candidate_comparison_before
        or receipt_comparison_before != parent_comparison_bytes
        or hashlib.sha256(receipt_comparison_before).hexdigest()
        != getattr(receipt, "comparison_before_sha256", None)
        or receipt_comparison_after != candidate_comparison_after
        or receipt_comparison_after != committed_comparison_bytes
        or hashlib.sha256(receipt_comparison_after).hexdigest()
        != getattr(receipt, "comparison_after_sha256", None)
        or len(receipt_comparison_after)
        != getattr(receipt, "comparison_after_bytes", None)
        or receipt_evidence != getattr(candidate, "evidence", None)
    ):
        _reject(code)
    return receipt_evidence


def authenticate_pushed_result(
    repo_root: Path,
    *,
    dependencies: PushedResultDependencies | None = None,
) -> dict[str, Any]:
    """Authenticate pushed R and independently replay its sealed private result.

    The gate is deliberately separate from the one-shot preflight and the
    effectful development worker.  It reads committed Git objects and the
    already-sealed private attempt, returns an ephemeral redacted receipt, and
    never writes an artifact or authorizes confirmation, final, or trading.
    """

    root = _canonical_repo_root(repo_root)
    deps = (
        dependencies
        if dependencies is not None
        else build_production_pushed_result_dependencies()
    )
    if type(deps) is not PushedResultDependencies:
        _reject("pushed_result_dependencies_invalid")
    store: Any | None = None
    try:
        # Complete public/Git authentication happens before contact or private
        # evidence is loaded.
        git_before = _inspect_result_git(root)
        parent = _classify_result_parent(root, git_before["parent"])
        result_payload = git_before["result_blob"]["payload"]
        result = _parse_canonical_mapping_bytes(
            result_payload,
            maximum=MAX_RESULT_ARTIFACT_BYTES,
            code="pushed_result_artifact_invalid",
        )
        result = _validate_public_result_basics(result, parent=parent)
        parent_comparison = _git_blob_material(
            root,
            git_before["parent"],
            contract.COMPARISON_PATH,
            maximum=MAX_COMPARISON_BYTES,
            code="pushed_result_comparison_parent_invalid",
        )["payload"]
        expected_comparison = deps.build_comparison_update(parent_comparison, result)
        if (
            type(expected_comparison) is not bytes
            or expected_comparison != git_before["comparison_blob"]["payload"]
        ):
            _reject("pushed_result_comparison_invalid")

        contact = deps.load_private_contact(root)
        if type(contact) is not str or not contact:
            _reject("pushed_result_private_contact_invalid")
        tokens = _privacy_token_bytes(
            root, lambda path: deps.privacy_tokens(path, contact)
        )
        _scan_public_bytes(result_payload, tokens)
        if parent["authorized_parent_kind"] == "continuation":
            _scan_public_bytes(parent["pause_blob"]["payload"], tokens)
            _scan_public_bytes(parent["continuation_blob"]["payload"], tokens)
        comparison_payload = git_before["comparison_blob"]["payload"]
        if _payload_contains_forbidden_token(comparison_payload, tokens):
            _reject("pushed_result_public_privacy_failed")
        inserted = _comparison_inserted_bytes(parent_comparison, comparison_payload)
        if not inserted:
            _reject("pushed_result_comparison_invalid")
        _scan_public_bytes(inserted, tokens)

        private_tree_tokens = _contact_v38_private_tokens(root, contact)
        v315_before = _snapshot_serialized_tree(
            root / Path(contract.PRIVATE_NAMESPACE),
            repo_root=root,
            forbidden=private_tree_tokens,
            code="pushed_result_private_privacy_failed",
        )
        v38_before = _snapshot_serialized_tree(
            root / V38_PRIVATE_ROOT,
            repo_root=root,
            forbidden=None,
            code="pushed_result_v38_inventory_invalid",
        )

        preflight_receipt = deps.authenticate_preflight_revision(
            root, parent["preflight_commit"]
        )
        if (
            not isinstance(preflight_receipt, Mapping)
            or preflight_receipt.get("preflight_commit")
            != parent["preflight_commit"]
            or preflight_receipt.get("pushed_preflight_gate_passed") is not True
            or preflight_receipt.get("development_authorized") is not False
        ):
            _reject("pushed_result_preflight_authority_invalid")
        preflight_receipt = _plain_json_mapping(
            preflight_receipt, code="pushed_result_preflight_authority_invalid"
        )
        result = _validate_public_result_basics(
            result, parent=parent, preflight_receipt=preflight_receipt
        )
        execution_authority = _store_execution_authority(preflight_receipt)

        source = deps.authenticate_source(root, contact)
        projection = deps.build_projection(source)
        projection_manifest_raw = getattr(projection, "manifest", None)
        projection_manifest = validate_bridge_manifest(projection_manifest_raw)
        if (
            projection_manifest["bridge_sha256"]
            != execution_authority["source"]["bridge_sha256"]
            or result.get("source_authority") != execution_authority["source"]
        ):
            _reject("pushed_result_source_authority_invalid")
        context = deps.rebuild_attempt_context(execution_authority, projection)
        if not isinstance(context, Mapping) or set(context) != {"authority", "plan"}:
            _reject("pushed_result_attempt_authority_invalid")
        authority = context["authority"]
        plan = context["plan"]
        if not isinstance(authority, Mapping) or dict(authority) != execution_authority:
            _reject("pushed_result_attempt_authority_invalid")
        authority = _plain_json_mapping(
            authority, code="pushed_result_attempt_authority_invalid"
        )

        store = deps.open_store(
            root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE), authority
        )
        snapshot = store.snapshot
        receipt = store.committed_terminal_evidence()
        candidate = getattr(store, "terminal_candidate", None)
        receipt_evidence = _validate_pushed_terminal_candidate_binding(
            result=result,
            snapshot=snapshot,
            receipt=receipt,
            candidate=candidate,
            committed_result_bytes=git_before["result_blob"]["payload"],
            parent_comparison_bytes=parent_comparison,
            committed_comparison_bytes=comparison_payload,
        )

        if parent["authorized_parent_kind"] == "continuation":
            continuation_document_sha256 = (
                _authenticate_result_continuation_prerequisites(
                    parent=parent,
                    dependencies=deps,
                    store=store,
                    authority=authority,
                    plan=plan,
                )
            )
            if (
                getattr(snapshot, "paused", None) is not True
                or getattr(snapshot, "continuation_authorized", None) is not True
                or getattr(snapshot, "continuation_sha256", None)
                != continuation_document_sha256
                or getattr(snapshot, "continuation_commit", None)
                != parent["continuation_commit"]
                or not _is_sha256(
                    getattr(snapshot, "continuation_permission_sha256", None)
                )
            ):
                _reject("pushed_result_continuation_binding_invalid")
        elif (
            getattr(snapshot, "paused", None) is True
            or getattr(snapshot, "continuation_authorized", None) is True
            or getattr(snapshot, "continuation_sha256", None) is not None
            or getattr(snapshot, "continuation_permission_sha256", None) is not None
            or getattr(snapshot, "continuation_commit", None) is not None
        ):
            _reject("pushed_result_continuation_binding_invalid")

        effect_report_raw = deps.build_effect_report(snapshot)
        completed_terminal = getattr(receipt, "status", None) == "completed"
        effect_report = _validate_effect_report(
            effect_report_raw,
            snapshot=snapshot,
            completed_terminal=completed_terminal,
            parent_kind=parent["authorized_parent_kind"],
        )
        public_effect = deps.build_public_effect_report(effect_report)
        if (
            not isinstance(public_effect, Mapping)
            or result.get("effect_report") != dict(public_effect)
        ):
            _reject("pushed_result_effects_invalid")
        private_material = _validate_private_terminal_material(
            receipt_evidence,
            receipt=receipt,
            snapshot=snapshot,
            authority=authority,
            projection_manifest=projection_manifest,
            parent=parent,
            effect_report=effect_report,
        )
        if completed_terminal:
            replayed = deps.replay_completed_terminal(
                store=store,
                authority=authority,
                projection=projection,
                plan=plan,
                effect_report=effect_report,
                invocation_parent=parent["authorized_parent"],
                invocation_parent_kind=parent["authorized_parent_kind"],
            )
        else:
            replayed = deps.replay_failure_terminal(
                snapshot=snapshot,
                receipt=receipt,
                authority=authority,
                effect_report=effect_report,
                invocation_parent=parent["authorized_parent"],
                invocation_parent_kind=parent["authorized_parent_kind"],
                evidence=private_material,
            )
        if not isinstance(replayed, Mapping) or dict(replayed) != private_material:
            _reject("pushed_result_independent_replay_failed")

        expected_public = deps.build_public_terminal_artifact(
            authority=authority, terminal_receipt=receipt
        )
        if not isinstance(expected_public, Mapping) or dict(expected_public) != result:
            _reject("pushed_result_public_replay_failed")
        snapshot_after_replay = store.snapshot
        if snapshot_after_replay != snapshot:
            _reject("pushed_result_store_changed")
        store.close()
        store = None

        # Re-scan both private authorities and Git after all private replay.
        # Equality detects any write or authority race while the gate ran.
        v315_after = _snapshot_serialized_tree(
            root / Path(contract.PRIVATE_NAMESPACE),
            repo_root=root,
            forbidden=private_tree_tokens,
            code="pushed_result_private_privacy_failed",
        )
        v38_after = _snapshot_serialized_tree(
            root / V38_PRIVATE_ROOT,
            repo_root=root,
            forbidden=None,
            code="pushed_result_v38_inventory_invalid",
        )
        if v315_after != v315_before:
            _reject("pushed_result_private_state_changed")
        if v38_after != v38_before:
            _reject("pushed_result_v38_changed")
        git_after = _inspect_result_git(root)
        if git_after != git_before:
            _reject("pushed_result_git_changed")

        ancestry = {
            "route": result["route"],
            "branch": git_before["branch"],
            "commit": git_before["commit"],
            "tree": git_before["tree"],
            "parent": git_before["parent"],
            "authorized_parent": parent["authorized_parent"],
            "authorized_parent_kind": parent["authorized_parent_kind"],
            "local_head": git_before["commit"],
            "remote_head": git_before["remote"],
            "changed_paths": git_before["changed_paths"],
            "clean_worktree": git_before["status"] == "",
            "terminal_sealed": True,
            "independent_replay_passed": True,
            "privacy_passed": True,
            "v38_unchanged": True,
        }
        try:
            contract.validate_result_ancestry(ancestry)
        except Exception:
            _reject("pushed_result_ancestry_invalid")

        science_summary = private_material.get("science_summary")
        gate_body = {
            "schema_version": PUSHED_RESULT_GATE_SCHEMA_VERSION,
            "status": "passed",
            "development_outcome": result["status"],
            "route": result["route"],
            "branch": git_before["branch"],
            "result_commit": git_before["commit"],
            "result_tree": git_before["tree"],
            "authorized_parent": parent["authorized_parent"],
            "authorized_parent_kind": parent["authorized_parent_kind"],
            "preflight_commit": parent["preflight_commit"],
            "result_git_blob_sha1": git_before["result_blob"]["git_blob_sha1"],
            "result_literal_sha256": git_before["result_blob"]["literal_sha256"],
            "development_result_sha256": result["development_result_sha256"],
            "comparison_git_blob_sha1": git_before["comparison_blob"][
                "git_blob_sha1"
            ],
            "comparison_literal_sha256": git_before["comparison_blob"][
                "literal_sha256"
            ],
            "terminal_candidate_sha256": getattr(receipt, "candidate_sha256"),
            "private_terminal_material_sha256": private_material[
                "private_terminal_material_sha256"
            ],
            "terminal_event_sha256": getattr(receipt, "event_sha256"),
            "effect_report_sha256": contract.canonical_sha256(effect_report),
            "science_summary_sha256": (
                contract.canonical_sha256(science_summary)
                if science_summary is not None
                else None
            ),
            "source_bridge_sha256": projection_manifest["bridge_sha256"],
            "private_namespace_inventory_sha256": v315_after[
                "inventory_sha256"
            ],
            "v38_inventory_snapshot_sha256": v38_after["inventory_sha256"],
            "gates": {
                "remote_commit_and_tree_authenticated": True,
                "exact_two_path_delta_authenticated": True,
                "git_blobs_and_canonical_hashes_authenticated": True,
                "private_terminal_seal_replayed": True,
                "effect_counts_replayed": True,
                "deterministic_science_replayed": completed_terminal,
                "failure_material_replayed": not completed_terminal,
                "public_projection_replayed": True,
                "privacy_passed": True,
                "v38_unchanged": True,
                "final_recheck_passed": True,
            },
            "confirmation_preregistration_eligible": result["status"] == "passed",
            "confirmation_execution_authorized": False,
            "final_execution_authorized": False,
            "real_money_authorized": False,
        }
        gate_receipt = {
            **gate_body,
            "pushed_result_gate_sha256": contract.canonical_sha256(gate_body),
        }
        _scan_public_bytes(contract.canonical_json_bytes(gate_receipt), tokens)
        return gate_receipt
    except V315PreflightError:
        raise
    except Exception:
        _reject("pushed_result_dependency_failed")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


_QUALIFICATION_MAX_LOG_BYTES: Final[int] = 128 * 1024 * 1024
_QUALIFICATION_MAX_XML_BYTES: Final[int] = 64 * 1024 * 1024
_QUALIFICATION_STREAM_CHUNK_BYTES: Final[int] = 64 * 1024


def _qualification_runtime_identity() -> dict[str, Any]:
    """Authenticate Python and pytest bytes without importing pytest here."""

    try:
        executable = Path(sys.executable).resolve(strict=True)
        executable_details = executable.lstat()
        install_paths = sysconfig.get_paths()
        purelib = Path(install_paths["purelib"]).resolve(strict=True)
        pytest_path = (purelib / "pytest" / "__init__.py").resolve(strict=True)
        pytest_details = pytest_path.lstat()
        executable_bytes = _read_regular_file(
            executable,
            maximum=None,
            code="preflight_qualification_runtime_invalid",
        )
        pytest_bytes = _read_regular_file(
            pytest_path,
            maximum=None,
            code="preflight_qualification_runtime_invalid",
        )
    except (AttributeError, KeyError, OSError, TypeError):
        _reject("preflight_qualification_runtime_invalid")
    if (
        sys.version != contract.QUALIFICATION_PYTHON_VERSION
        or sys.implementation.cache_tag != contract.QUALIFICATION_PYTHON_CACHE_TAG
        or os.name != contract.QUALIFICATION_OS_NAME
        or sys.platform != contract.QUALIFICATION_SYS_PLATFORM
        or executable.name != contract.QUALIFICATION_EXECUTABLE_BASENAME
        or not stat.S_ISREG(executable_details.st_mode)
        or stat.S_ISLNK(executable_details.st_mode)
        or len(executable_bytes) != contract.QUALIFICATION_EXECUTABLE_BYTES
        or hashlib.sha256(executable_bytes).hexdigest()
        != contract.QUALIFICATION_EXECUTABLE_SHA256
        or pytest_path != purelib / "pytest" / "__init__.py"
        or not stat.S_ISREG(pytest_details.st_mode)
        or stat.S_ISLNK(pytest_details.st_mode)
        or getattr(pytest_details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or len(pytest_bytes) != contract.QUALIFICATION_PYTEST_INIT_BYTES
        or hashlib.sha256(pytest_bytes).hexdigest()
        != contract.QUALIFICATION_PYTEST_INIT_SHA256
    ):
        _reject("preflight_qualification_runtime_invalid")
    return {
        "python_version": sys.version,
        "python_cache_tag": sys.implementation.cache_tag,
        "os_name": os.name,
        "sys_platform": sys.platform,
        "executable_path": str(executable),
        "executable_basename": executable.name,
        "executable_bytes": len(executable_bytes),
        "executable_sha256": hashlib.sha256(executable_bytes).hexdigest(),
        "pytest_version": contract.QUALIFICATION_PYTEST_VERSION,
        "pytest_init_path": str(pytest_path),
        "pytest_init_bytes": len(pytest_bytes),
        "pytest_init_sha256": hashlib.sha256(pytest_bytes).hexdigest(),
    }


def _qualification_environment(
    runtime: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Construct the exact cleared 16-name qualification environment."""

    try:
        trusted = _runtime_active_preflight_context().get("trusted_git")
        if (
            isinstance(trusted, tuple)
            and len(trusted) == 3
            and isinstance(trusted[0], Path)
        ):
            git_executable = trusted[0]
        else:
            git_executable, _git_bytes, _git_version = (
                _resolve_authenticated_git_runtime()
            )
        reduced = build_reduced_runtime_environment(
            os.environ,
            git_executable=git_executable,
            python_executable=runtime["executable_path"],
        )
        environment = dict(reduced["child_environment"])
        profile = dict(reduced["process_environment_material"])
        for name, value in contract.QUALIFICATION_ENVIRONMENT:
            environment[name] = value
            profile[name] = value
    except V315PreflightError:
        raise
    except Exception:
        _reject("preflight_qualification_environment_invalid")
    if (
        set(environment) != set(contract.QUALIFICATION_ENVIRONMENT_NAMES)
        or len(environment) != len(contract.QUALIFICATION_ENVIRONMENT_NAMES)
        or any(
            environment.get(name) != value
            for name, value in contract.QUALIFICATION_ENVIRONMENT
        )
        or any(
            not environment.get(name)
            for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH")
        )
    ):
        _reject("preflight_qualification_environment_invalid")
    return environment, {
        name: profile[name]
        for name in sorted(profile, key=lambda value: value.encode("utf-8"))
    }


def _qualification_argv(
    root: Path,
    *,
    runtime: Mapping[str, Any],
    junit_partial: Path,
    phase: str,
    mode: str,
) -> list[str]:
    if mode not in contract.QUALIFICATION_MODES:
        _reject("preflight_qualification_mode_invalid")
    try:
        canonical_root = root.resolve(strict=True)
    except OSError:
        _reject("preflight_qualification_path_invalid")
    dispatch = [
        canonical_root.as_posix()
        if token == contract.QUALIFICATION_CANONICAL_ROOT_TOKEN
        else token
        for token in contract.QUALIFICATION_DISPATCH_PREFIX
    ]
    # The bootstrap compares Windows paths through normcase/normpath.  Keep the
    # native absolute spelling in argv so its samefile check is unambiguous.
    dispatch[2] = str(canonical_root)
    argv = [
        str(runtime["executable_path"]),
        *contract.QUALIFICATION_PYTHON_FLAGS,
        "-c",
        contract.QUALIFICATION_BOOTSTRAP_LITERAL,
        *dispatch,
        *contract.QUALIFICATION_PYTEST_ARGS,
        f"--junitxml={junit_partial}",
    ]
    if mode == "collection":
        argv.append("--collect-only")
    argv.extend(_qualification_selectors(phase))
    return argv


def _qualification_terminate_process_tree(process: Any) -> bool:
    """Terminate the exact Windows pytest process tree and reap its parent."""

    try:
        if process.poll() is not None:
            return True
        pid = process.pid
    except Exception:
        return False
    if type(pid) is not int or pid <= 0 or sys.platform != "win32":
        return False
    try:
        completed = subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30.0,
        )
        if completed.returncode != 0 and process.poll() is None:
            return False
        process.wait(timeout=30.0)
        return process.poll() is not None
    except (OSError, subprocess.SubprocessError):
        return False


class _QualificationWindowsJob:
    """One kill-on-close Windows Job Object containing a pytest process tree."""

    def __init__(self, handle: int, kernel32: Any) -> None:
        self._handle: int | None = handle
        self._kernel32 = kernel32

    def terminate(self) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            return bool(self._kernel32.TerminateJobObject(handle, 1))
        except Exception:
            return False

    def active_process_count(self) -> int | None:
        handle = self._handle
        if handle is None:
            return None
        try:
            from ctypes import wintypes

            class BasicAccountingInformation(ctypes.Structure):
                _fields_ = [
                    ("TotalUserTime", ctypes.c_longlong),
                    ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                    ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", wintypes.DWORD),
                    ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD),
                    ("TotalTerminatedProcesses", wintypes.DWORD),
                ]

            information = BasicAccountingInformation()
            returned = wintypes.DWORD()
            if not self._kernel32.QueryInformationJobObject(
                handle,
                1,
                ctypes.byref(information),
                ctypes.sizeof(information),
                ctypes.byref(returned),
            ):
                return None
            return int(information.ActiveProcesses)
        except Exception:
            return None

    def close(self) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            closed = bool(self._kernel32.CloseHandle(handle))
        except Exception:
            return False
        if closed:
            self._handle = None
        return closed

    def __del__(self) -> None:
        if self._handle is not None:
            self.terminate()
            self.close()


def _qualification_create_windows_job(
    process: Any,
) -> _QualificationWindowsJob | None:
    """Immediately bind a new pytest process to a kill-on-close Job Object."""

    if sys.platform != "win32":
        return None
    try:
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        job = _QualificationWindowsJob(handle, kernel32)
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            job.close()
            return None
        process_handle = wintypes.HANDLE(int(process._handle))
        if not kernel32.AssignProcessToJobObject(handle, process_handle):
            job.close()
            return None
        return job
    except Exception:
        return None


def _qualification_resume_windows_process(process: Any) -> bool:
    """Resume a CREATE_SUSPENDED child only after Job assignment succeeds."""

    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes

        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        return int(ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle)))) == 0
    except Exception:
        return False


def _qualification_kill_suspended_process(process: Any) -> bool:
    """Kill and reap a direct child that has never been allowed to execute."""

    try:
        process.kill()
        process.wait(timeout=30.0)
        return process.poll() is not None
    except (OSError, subprocess.SubprocessError):
        return _qualification_terminate_process_tree(process)


def _qualification_terminate_job_and_reap(
    job: _QualificationWindowsJob,
    process: Any,
) -> tuple[bool, int | None]:
    """Terminate all members, reap the parent, prove zero active, then close."""

    terminated = job.terminate()
    exit_code: int | None = None
    reaped = False
    try:
        exit_code = process.wait(timeout=30.0)
        reaped = process.poll() is not None
    except (OSError, subprocess.SubprocessError):
        reaped = False
    quiescent = False
    stop = time.monotonic() + 30.0
    while time.monotonic() < stop:
        active = job.active_process_count()
        if active == 0:
            quiescent = True
            break
        if active is None:
            break
        time.sleep(0.01)
    closed = job.close()
    return terminated and reaped and quiescent and closed, exit_code


def _qualification_selectors(phase: str) -> tuple[str, ...]:
    if phase == contract.QUALIFICATION_PHASE_V315:
        return contract.QUALIFICATION_LATEST_TEST_PATHS
    if phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
        return contract.QUALIFICATION_SHARED_TEST_PATHS
    if phase == contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY:
        return (contract.QUALIFICATION_SENTINEL_NODE_ID,)
    _reject("preflight_qualification_phase_invalid")


def _qualification_require_before_deadline(deadline_monotonic: float) -> None:
    try:
        now = time.monotonic()
    except Exception:
        _reject("preflight_qualification_clock_invalid")
    if (
        type(now) is not float
        or not 0.0 <= now < deadline_monotonic
    ):
        _reject("preflight_qualification_timeout")


def _qualification_write_new(path: Path, payload: bytes, *, code: str) -> None:
    """Flush, no-replace rename, reopen, and byte-verify one private payload."""

    if type(payload) is not bytes:
        _reject(code)
    pending = path.with_name(path.name + ".v315-pending")
    if (
        _optional_lstat(path, code=code) is not None
        or _optional_lstat(pending, code=code) is not None
    ):
        _reject(code)
    try:
        with pending.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        if _optional_lstat(path, code=code) is not None:
            _reject(code)
        os.rename(pending, path)
        _fsync_directory(path.parent)
    except V315PreflightError:
        raise
    except OSError:
        _reject(code)
    _assert_exact_regular_file(path, payload, code=code)
    if _optional_lstat(pending, code=code) is not None:
        _reject(code)


def _qualification_promote_partial(
    partial: Path,
    final: Path,
    *,
    maximum: int,
    code: str,
) -> bytes:
    if _optional_lstat(final, code=code) is not None:
        _reject(code)
    try:
        with partial.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial.parent)
    except OSError:
        _reject(code)
    payload = _read_regular_file(partial, maximum=maximum, code=code)
    try:
        os.rename(partial, final)
        _fsync_directory(final.parent)
    except OSError:
        _reject(code)
    _assert_exact_regular_file(final, payload, code=code)
    if _optional_lstat(partial, code=code) is not None:
        _reject(code)
    return payload


def _qualification_collection_evidence(stdout: bytes) -> tuple[int, str]:
    nodes = _qualification_collection_node_ids(stdout)
    payload = "\n".join(nodes).encode("utf-8")
    return len(nodes), hashlib.sha256(payload).hexdigest()


def _qualification_collection_node_ids(stdout: bytes) -> list[str]:
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeError:
        _reject("preflight_qualification_collection_invalid")
    nodes = [
        line
        for line in text.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if not nodes:
        _reject("preflight_qualification_collection_invalid")
    return nodes


def _qualification_executed_node_evidence(
    payload: bytes,
    sealed_node_ids: Sequence[str],
) -> tuple[int, str, list[str]]:
    """Map xunit2 testcase identities back to exact collected pytest node IDs."""

    if (
        not isinstance(sealed_node_ids, Sequence)
        or isinstance(sealed_node_ids, (str, bytes, bytearray))
        or any(type(node_id) is not str for node_id in sealed_node_ids)
    ):
        _reject("preflight_qualification_execution_nodes_invalid")
    expected = Counter(sealed_node_ids)
    module_paths: dict[str, str] = {}
    for node_id in sealed_node_ids:
        parts = node_id.split("::")
        module_path = parts[0]
        if (
            len(parts) < 2
            or not module_path.startswith("tests/")
            or not module_path.endswith(".py")
            or any(not part for part in parts)
        ):
            _reject("preflight_qualification_execution_nodes_invalid")
        module_name = module_path[:-3].replace("/", ".")
        prior_path = module_paths.setdefault(module_name, module_path)
        if prior_path != module_path:
            _reject("preflight_qualification_execution_nodes_invalid")
    try:
        xml_root = ET.fromstring(payload)
    except ET.ParseError:
        _reject("preflight_qualification_execution_nodes_invalid")
    observed: list[str] = []
    for case in xml_root.iter("testcase"):
        classname = case.get("classname")
        name = case.get("name")
        if type(classname) is not str or not classname or type(name) is not str or not name:
            _reject("preflight_qualification_execution_nodes_invalid")
        candidates = [
            module_name
            for module_name in module_paths
            if classname == module_name
            or classname.startswith(module_name + ".")
        ]
        if not candidates:
            _reject("preflight_qualification_execution_nodes_invalid")
        longest = max(len(module_name) for module_name in candidates)
        longest_candidates = [
            module_name
            for module_name in candidates
            if len(module_name) == longest
        ]
        if len(longest_candidates) != 1:
            _reject("preflight_qualification_execution_nodes_invalid")
        module_name = longest_candidates[0]
        class_suffix = classname[len(module_name) :]
        if class_suffix and not class_suffix.startswith("."):
            _reject("preflight_qualification_execution_nodes_invalid")
        class_parts = class_suffix[1:].split(".") if class_suffix else []
        if any(not part for part in class_parts):
            _reject("preflight_qualification_execution_nodes_invalid")
        node_id = "::".join(
            [module_paths[module_name], *class_parts, name]
        )
        if expected[node_id] <= 0:
            _reject("preflight_qualification_execution_nodes_invalid")
        observed.append(node_id)
    if (
        not observed
        or len(observed) != len(sealed_node_ids)
        or Counter(observed) != expected
    ):
        _reject("preflight_qualification_execution_nodes_invalid")
    encoded = "\n".join(observed).encode("utf-8")
    return len(observed), hashlib.sha256(encoded).hexdigest(), observed


_PYTEST_SUMMARY_ITEM_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<count>[0-9]+)\s+(?P<label>passed|failed|errors?|skipped|"
    r"xfailed|xpassed|warnings?|deselected)\Z"
)
_PYTEST_SUMMARY_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*=*\s*(?P<body>.+?)\s+in\s+[0-9]+(?:\.[0-9]+)?s"
    r"(?: \([0-9]+:[0-5][0-9]:[0-5][0-9]\))?\s*=*\s*$"
)


def _qualification_junit_counts(
    payload: bytes,
    stdout: bytes,
    *,
    require_terminal_summary: bool = True,
) -> dict[str, int]:
    try:
        xml_root = ET.fromstring(payload)
    except ET.ParseError:
        _reject("preflight_qualification_junit_invalid")
    cases = list(xml_root.iter("testcase"))
    failed = 0
    errors = 0
    skipped = 0
    xfailed = 0
    for case in cases:
        if case.find("failure") is not None:
            failed += 1
        if case.find("error") is not None:
            errors += 1
        skipped_node = case.find("skipped")
        if skipped_node is not None:
            if skipped_node.get("type") == "pytest.xfail":
                xfailed += 1
            else:
                skipped += 1
    xpassed = 0
    summary_counts: dict[str, int] | None = None
    if require_terminal_summary:
        try:
            summary = stdout.decode("utf-8", errors="strict")
        except UnicodeError:
            _reject("preflight_qualification_junit_invalid")
        records: list[dict[str, int]] = []
        for line in summary.splitlines():
            match = _PYTEST_SUMMARY_LINE_RE.fullmatch(line)
            if match is None:
                continue
            parsed = {
                "passed": 0,
                "failed": 0,
                "error": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
            }
            seen: set[str] = set()
            valid = True
            for item in match.group("body").split(", "):
                item_match = _PYTEST_SUMMARY_ITEM_RE.fullmatch(item.strip())
                if item_match is None:
                    valid = False
                    break
                label = item_match.group("label")
                normalized = {
                    "errors": "error",
                    "warnings": "warning",
                }.get(label, label)
                if normalized in seen:
                    valid = False
                    break
                seen.add(normalized)
                if normalized in parsed:
                    parsed[normalized] = int(item_match.group("count"))
            if valid and seen & set(parsed):
                records.append(parsed)
        if len(records) != 1:
            _reject("preflight_qualification_junit_invalid")
        summary_counts = records[0]
        xpassed = summary_counts["xpassed"]
    passed = len(cases) - failed - errors - skipped - xfailed - xpassed
    if passed < 0:
        _reject("preflight_qualification_junit_invalid")
    counts = {
        "passed_count": passed,
        "failed_count": failed,
        "error_count": errors,
        "skipped_count": skipped,
        "xfailed_count": xfailed,
        "xpassed_count": xpassed,
    }
    if summary_counts is not None and (
        summary_counts["passed"] != counts["passed_count"]
        or summary_counts["failed"] != counts["failed_count"]
        or summary_counts["error"] != counts["error_count"]
        or summary_counts["skipped"] != counts["skipped_count"]
        or summary_counts["xfailed"] != counts["xfailed_count"]
        or summary_counts["xpassed"] != counts["xpassed_count"]
    ):
        _reject("preflight_qualification_junit_invalid")
    return counts


def _qualification_completion_receipt(
    *,
    phase: str,
    mode: str,
    result: Mapping[str, Any],
    result_literal_sha256: str,
    finalization_overrun: bool,
) -> dict[str, Any]:
    status = (
        "failed"
        if result["status"] != "passed" or finalization_overrun
        else "completed"
    )
    terminal_reason = (
        "timeout" if finalization_overrun else result["terminal_reason"]
    )
    return _self_hash(
        {
            "schema_version": QUALIFICATION_COMPLETION_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "phase": phase,
            "mode": mode,
            "status": status,
            "terminal_reason": terminal_reason,
            "deadline_overrun": (
                bool(result["deadline_overrun"]) or finalization_overrun
            ),
            "result_sha256": result["result_sha256"],
            "result_literal_sha256": result_literal_sha256,
            "log_sha256": result["log_sha256"],
            "stdout_sha256": result["stdout_sha256"],
            "xml_sha256": result["xml_sha256"],
        },
        "completion_sha256",
    )


def _qualification_run_process(
    root: Path,
    *,
    private_root: Path,
    phase: str,
    mode: str,
    deadline_monotonic: float,
    runtime: Mapping[str, Any],
    repository_commit: str,
    repository_tree: str,
    clean_state_sha256: str,
    sealed_collection: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selectors = _qualification_selectors(phase)
    if mode not in contract.QUALIFICATION_MODES:
        _reject("preflight_qualification_mode_invalid")
    mode_root = private_root / "qualification" / phase / mode
    junit_partial = private_root / Path(
        contract.QUALIFICATION_JUNIT_RELATIVE_TEMPLATE.format(
            phase=phase,
            mode=mode,
        )
    )
    if junit_partial.parent != mode_root:
        _reject("preflight_qualification_path_invalid")
    log_partial = mode_root / "output.partial.log"
    log_final = mode_root / "output.log"
    stdout_final = mode_root / "stdout.bin"
    junit_final = mode_root / "junit.xml"
    intent_path = mode_root / "intent.json"
    result_path = mode_root / "result.json"
    completion_path = mode_root / "complete.json"
    environment, environment_profile = _qualification_environment(runtime)
    argv = _qualification_argv(
        root,
        runtime=runtime,
        junit_partial=junit_partial,
        phase=phase,
        mode=mode,
    )
    intent = _self_hash(
        {
            "schema_version": contract.QUALIFICATION_INTENT_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "phase": phase,
            "mode": mode,
            "argv": argv,
            "environment_profile": environment_profile,
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "clean_state_sha256": clean_state_sha256,
            "deadline_monotonic_ns": int(deadline_monotonic * 1_000_000_000),
            "runtime": copy.deepcopy(dict(runtime)),
            "collection_manifest_sha256": (
                sealed_collection["collection_manifest_sha256"]
                if mode == "execution" and sealed_collection is not None
                else None
            ),
        },
        "intent_sha256",
    )
    _qualification_require_before_deadline(deadline_monotonic)
    _qualification_write_new(
        intent_path,
        contract.canonical_json_bytes(intent),
        code="preflight_qualification_intent_unwritable",
    )
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow = [False]
    drain_errors: list[str] = []
    log_lock = threading.Lock()
    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    process: Any = None
    job: _QualificationWindowsJob | None = None
    process_resumed = False
    containment_proven = True
    exit_code: int | None = None
    timed_out = False
    exception_code: str | None = None
    try:
        with log_partial.open("xb") as log_handle:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    creationflags=int(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
                    ),
                )
            except (OSError, subprocess.SubprocessError):
                exception_code = "spawn_failed"
            if process is not None and containment_proven:
                job = _qualification_create_windows_job(process)
                if job is None:
                    exception_code = "containment_failed"
                    containment_proven = _qualification_kill_suspended_process(
                        process
                    )
                elif not _qualification_resume_windows_process(process):
                    exception_code = "resume_failed"
                    containment_proven, exit_code = (
                        _qualification_terminate_job_and_reap(job, process)
                    )
                    job = None
                else:
                    process_resumed = True

            def drain(pipe: Any, target: bytearray) -> None:
                try:
                    while True:
                        chunk = pipe.read(_QUALIFICATION_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        if type(chunk) is not bytes:
                            drain_errors.append("non_binary_output")
                            continue
                        if len(target) + len(chunk) <= _QUALIFICATION_MAX_LOG_BYTES:
                            target.extend(chunk)
                        else:
                            overflow[0] = True
                        with log_lock:
                            log_handle.write(chunk)
                            log_handle.flush()
                            os.fsync(log_handle.fileno())
                except Exception:
                    drain_errors.append("drain_failed")

            threads: list[threading.Thread] = []
            if process is not None and containment_proven:
                for pipe, target in (
                    (process.stdout, stdout_buffer),
                    (process.stderr, stderr_buffer),
                ):
                    thread = threading.Thread(
                        target=drain,
                        args=(pipe, target),
                        daemon=True,
                    )
                    threads.append(thread)
                    thread.start()
                while job is not None and process_resumed:
                    try:
                        if process.poll() is not None:
                            break
                        remaining = deadline_monotonic - time.monotonic()
                        if remaining <= 0.0:
                            timed_out = True
                            break
                        process.wait(timeout=min(0.2, remaining))
                    except subprocess.TimeoutExpired:
                        continue
                    except (OSError, subprocess.SubprocessError):
                        exception_code = "wait_failed"
                        break
                if job is not None:
                    containment_proven, exit_code = (
                        _qualification_terminate_job_and_reap(job, process)
                    )
                    job = None
                    if not containment_proven:
                        exception_code = "containment_cleanup_failed"
                elif containment_proven and exit_code is None:
                    try:
                        exit_code = process.wait(timeout=30.0)
                    except (OSError, subprocess.SubprocessError):
                        containment_proven = False
                for thread in threads:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        exception_code = exception_code or "drain_incomplete"
            log_handle.flush()
            os.fsync(log_handle.fileno())
    except FileExistsError:
        _reject("preflight_qualification_path_collision")
    except OSError:
        _reject("preflight_qualification_log_unwritable")
    if not containment_proven:
        _reject("preflight_qualification_containment_unproven")
    ended_monotonic_ns = time.monotonic_ns()
    ended_unix_ns = time.time_ns()
    if drain_errors:
        exception_code = exception_code or drain_errors[0]
    if overflow[0]:
        exception_code = exception_code or "output_oversized"
    log_payload = _qualification_promote_partial(
        log_partial,
        log_final,
        maximum=_QUALIFICATION_MAX_LOG_BYTES * 2,
        code="preflight_qualification_log_unwritable",
    )
    stdout_payload = bytes(stdout_buffer)
    _qualification_write_new(
        stdout_final,
        stdout_payload,
        code="preflight_qualification_stdout_unwritable",
    )
    raw_xml_payload: bytes | None = None
    if _optional_lstat(junit_partial, code="preflight_qualification_junit_invalid") is not None:
        raw_xml_payload = _qualification_promote_partial(
            junit_partial,
            junit_final,
            maximum=_QUALIFICATION_MAX_XML_BYTES,
            code="preflight_qualification_junit_invalid",
        )
    deadline_overrun = time.monotonic() >= deadline_monotonic
    xml_payload: bytes | None = None
    node_count: int | None = None
    node_list_sha256: str | None = None
    node_ids: list[str] | None = None
    counts: dict[str, int | None] = {
        "passed_count": None,
        "failed_count": None,
        "error_count": None,
        "skipped_count": None,
        "xfailed_count": None,
        "xpassed_count": None,
    }
    if raw_xml_payload is not None:
        try:
            ET.fromstring(raw_xml_payload)
        except ET.ParseError:
            exception_code = exception_code or "junit_evidence_invalid"
        else:
            if mode == "collection":
                xml_payload = raw_xml_payload
                try:
                    node_ids = _qualification_collection_node_ids(
                        stdout_payload
                    )
                    encoded_nodes = "\n".join(node_ids).encode("utf-8")
                    node_count = len(node_ids)
                    node_list_sha256 = hashlib.sha256(
                        encoded_nodes
                    ).hexdigest()
                    counts = _qualification_junit_counts(
                        raw_xml_payload,
                        stdout_payload,
                        require_terminal_summary=False,
                    )
                except V315PreflightError:
                    exception_code = exception_code or (
                        "collection_evidence_invalid"
                    )
            elif sealed_collection is not None:
                try:
                    (
                        node_count,
                        node_list_sha256,
                        node_ids,
                    ) = _qualification_executed_node_evidence(
                        raw_xml_payload,
                        sealed_collection["ordered_node_ids"],
                    )
                    counts = _qualification_junit_counts(
                        raw_xml_payload,
                        stdout_payload,
                    )
                    xml_payload = raw_xml_payload
                except (KeyError, TypeError, V315PreflightError):
                    node_count = None
                    node_list_sha256 = None
                    node_ids = None
                    counts = {field: None for field in counts}
                    exception_code = exception_code or (
                        "junit_evidence_invalid"
                    )
            else:
                exception_code = exception_code or (
                    "sealed_collection_or_xml_missing"
                )
    elif exception_code is None:
        exception_code = "sealed_collection_or_xml_missing"
    status = "passed"
    if (
        exception_code is not None
        or timed_out
        or deadline_overrun
        or exit_code != 0
        or xml_payload is None
        or node_count is None
        or node_list_sha256 is None
        or node_ids is None
    ):
        status = "failed"
    if mode == "collection" and status == "passed":
        if any(counts[field] != 0 for field in _QUALIFICATION_COUNT_FIELDS):
            status = "failed"
        elif phase == contract.QUALIFICATION_PHASE_V315:
            status = (
                "passed"
                if node_count >= contract.QUALIFICATION_LATEST_MIN_NODE_COUNT
                else "failed"
            )
        elif phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
            status = (
                "passed"
                if (
                    node_count == contract.QUALIFICATION_SHARED_NODE_COUNT
                    and node_list_sha256
                    == contract.QUALIFICATION_SHARED_NODE_LIST_SHA256
                )
                else "failed"
            )
        else:
            status = (
                "passed"
                if (
                    node_count == contract.QUALIFICATION_SENTINEL_NODE_COUNT
                    and node_list_sha256
                    == contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
                )
                else "failed"
            )
    if mode == "execution" and status == "passed":
        status = (
            "passed"
            if (
                sealed_collection is not None
                and node_count == sealed_collection["node_count"]
                and node_list_sha256
                == sealed_collection["node_list_sha256"]
                and node_ids == sealed_collection["ordered_node_ids"]
                and
                counts["passed_count"] == node_count
                and all(
                    counts[field] == 0
                    for field in (
                        "failed_count",
                        "error_count",
                        "skipped_count",
                        "xfailed_count",
                        "xpassed_count",
                    )
                )
            )
            else "failed"
        )
    if timed_out or deadline_overrun:
        terminal_reason = "timeout"
    elif exception_code == "spawn_failed":
        terminal_reason = "launch_error"
    elif exception_code is not None:
        terminal_reason = "exception"
    else:
        terminal_reason = "completed"
    result = _self_hash(
        {
            "schema_version": contract.QUALIFICATION_RESULT_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "phase": phase,
            "mode": mode,
            "status": status,
            "intent_sha256": intent["intent_sha256"],
            "collection_manifest_sha256": intent[
                "collection_manifest_sha256"
            ],
            "started_unix_ns": started_unix_ns,
            "ended_unix_ns": ended_unix_ns,
            "duration_monotonic_ns": max(
                0, ended_monotonic_ns - started_monotonic_ns
            ),
            "node_count": node_count,
            "node_list_sha256": node_list_sha256,
            "node_ids": node_ids,
            **counts,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "deadline_overrun": deadline_overrun,
            "terminal_reason": terminal_reason,
            "exception_code": exception_code,
            "log_bytes": len(log_payload),
            "log_sha256": hashlib.sha256(log_payload).hexdigest(),
            "stdout_bytes": len(stdout_payload),
            "stdout_sha256": hashlib.sha256(stdout_payload).hexdigest(),
            "xml_present": xml_payload is not None,
            "xml_bytes": len(xml_payload) if xml_payload is not None else None,
            "xml_sha256": (
                hashlib.sha256(xml_payload).hexdigest()
                if xml_payload is not None
                else None
            ),
        },
        "result_sha256",
    )
    result_payload = contract.canonical_json_bytes(result)
    _qualification_write_new(
        result_path,
        result_payload,
        code="preflight_qualification_result_unwritable",
    )
    finalization_overrun = time.monotonic() >= deadline_monotonic
    completion = _qualification_completion_receipt(
        phase=phase,
        mode=mode,
        result=result,
        result_literal_sha256=hashlib.sha256(result_payload).hexdigest(),
        finalization_overrun=finalization_overrun,
    )
    completion_status = completion["status"]
    _qualification_write_new(
        completion_path,
        contract.canonical_json_bytes(completion),
        code="preflight_qualification_completion_unwritable",
    )
    after_completion_overrun = time.monotonic() >= deadline_monotonic
    if after_completion_overrun and completion_status == "completed":
        deadline_failure = _self_hash(
            {
                "schema_version": QUALIFICATION_COMPLETION_SCHEMA_VERSION,
                "suite_identity": contract.QUALIFICATION_SUITE_ID,
                "phase": phase,
                "mode": mode,
                "status": "failed",
                "terminal_reason": "timeout",
                "deadline_overrun": True,
                "result_sha256": result["result_sha256"],
                "completion_sha256": completion["completion_sha256"],
            },
            "deadline_failure_sha256",
        )
        _qualification_write_new(
            mode_root / "deadline-failure.json",
            contract.canonical_json_bytes(deadline_failure),
            code="preflight_qualification_completion_unwritable",
        )
    if completion_status != "completed" or after_completion_overrun:
        _reject(
            "preflight_qualification_timeout"
            if (
                deadline_overrun
                or timed_out
                or finalization_overrun
                or after_completion_overrun
            )
            else "preflight_qualification_failed"
        )
    return result, completion


def _run_frozen_qualification_suite(root: Path) -> dict[str, Any]:
    """Run only the frozen three-phase qualification under one deadline."""

    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    qualification_root = private_root / "qualification"
    try:
        private_details = private_root.lstat()
        if (
            not stat.S_ISDIR(private_details.st_mode)
            or stat.S_ISLNK(private_details.st_mode)
            or getattr(private_details, "st_file_attributes", 0)
            & _REPARSE_ATTRIBUTE
        ):
            _reject("preflight_qualification_private_root_invalid")
        qualification_root.mkdir(exist_ok=False)
        for phase in contract.QUALIFICATION_PHASES:
            phase_root = qualification_root / phase
            phase_root.mkdir(exist_ok=False)
            for mode in contract.QUALIFICATION_MODES:
                (phase_root / mode).mkdir(exist_ok=False)
    except V315PreflightError:
        raise
    except (FileExistsError, OSError):
        _reject("preflight_qualification_path_collision")
    active_deadline = _runtime_active_preflight_context().get(
        "deadline_monotonic"
    )
    if active_deadline is None:
        deadline_monotonic = (
            time.monotonic() + contract.QUALIFICATION_TIMEOUT_SECONDS
        )
    elif type(active_deadline) is float and active_deadline > 0.0:
        deadline_monotonic = active_deadline
    else:
        _reject("preflight_qualification_clock_invalid")
    runtime = _qualification_runtime_identity()
    repository_commit = _git(root, "rev-parse", "HEAD")
    repository_tree = _git(root, "rev-parse", "HEAD^{tree}")
    clean_state = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if (
        type(repository_commit) is not str
        or type(repository_tree) is not str
        or type(clean_state) is not str
        or clean_state != ""
    ):
        _reject("preflight_qualification_repository_invalid")
    clean_state_sha256 = hashlib.sha256(b"").hexdigest()
    phase_results: list[dict[str, Any]] = []
    for phase in contract.QUALIFICATION_PHASES:
        _qualification_require_before_deadline(deadline_monotonic)
        collection, collection_completion = _qualification_run_process(
            root,
            private_root=private_root,
            phase=phase,
            mode="collection",
            deadline_monotonic=deadline_monotonic,
            runtime=runtime,
            repository_commit=repository_commit,
            repository_tree=repository_tree,
            clean_state_sha256=clean_state_sha256,
            sealed_collection=None,
        )
        if collection["status"] != "passed":
            _reject("preflight_qualification_failed")
        collection_manifest = _self_hash(
            {
                "schema_version": (
                    contract.QUALIFICATION_COLLECTION_SCHEMA_VERSION
                ),
                "suite_identity": contract.QUALIFICATION_SUITE_ID,
                "phase": phase,
                "status": "passed",
                "ordered_node_ids": collection["node_ids"],
                "node_count": collection["node_count"],
                "node_list_sha256": collection["node_list_sha256"],
                "collection_result_sha256": collection["result_sha256"],
                "collection_completion_sha256": collection_completion[
                    "completion_sha256"
                ],
            },
            "collection_manifest_sha256",
        )
        _qualification_write_new(
            qualification_root / phase / "collection-manifest.json",
            contract.canonical_json_bytes(collection_manifest),
            code="preflight_qualification_collection_unwritable",
        )
        _qualification_require_before_deadline(deadline_monotonic)
        execution, execution_completion = _qualification_run_process(
            root,
            private_root=private_root,
            phase=phase,
            mode="execution",
            deadline_monotonic=deadline_monotonic,
            runtime=runtime,
            repository_commit=repository_commit,
            repository_tree=repository_tree,
            clean_state_sha256=clean_state_sha256,
            sealed_collection=collection_manifest,
        )
        if execution["status"] != "passed":
            _reject("preflight_qualification_failed")
        phase_results.append(
            {
                "phase": phase,
                "node_count": collection["node_count"],
                "node_list_sha256": collection["node_list_sha256"],
                "collection_manifest_sha256": collection_manifest[
                    "collection_manifest_sha256"
                ],
                "collection_completion_sha256": collection_completion[
                    "completion_sha256"
                ],
                "execution_completion_sha256": execution_completion[
                    "completion_sha256"
                ],
                "collection": collection,
                "execution": execution,
            }
        )
    _qualification_require_before_deadline(deadline_monotonic)
    runtime_public = {
        key: runtime[key]
        for key in (
            "python_version",
            "python_cache_tag",
            "os_name",
            "sys_platform",
            "executable_basename",
            "executable_bytes",
            "executable_sha256",
            "pytest_version",
            "pytest_init_bytes",
            "pytest_init_sha256",
        )
    }
    private_aggregate = _self_hash(
        {
            "schema_version": (
                contract.QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION
            ),
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "status": "passed",
            "deadline_monotonic_ns": int(deadline_monotonic * 1_000_000_000),
            "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
            "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "clean_state_sha256": clean_state_sha256,
            "runtime": runtime,
            "phases": phase_results,
        },
        "private_aggregate_sha256",
    )
    aggregate_payload = contract.canonical_json_bytes(private_aggregate)
    _qualification_write_new(
        qualification_root / "private-aggregate.json",
        aggregate_payload,
        code="preflight_qualification_aggregate_unwritable",
    )
    aggregate_finalization_overrun = time.monotonic() >= deadline_monotonic
    aggregate_completion = _self_hash(
        {
            "schema_version": QUALIFICATION_COMPLETION_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "phase": "aggregate",
            "mode": "aggregate",
            "status": (
                "failed" if aggregate_finalization_overrun else "completed"
            ),
            "terminal_reason": (
                "timeout" if aggregate_finalization_overrun else "completed"
            ),
            "deadline_overrun": aggregate_finalization_overrun,
            "result_sha256": private_aggregate["private_aggregate_sha256"],
            "result_literal_sha256": hashlib.sha256(
                aggregate_payload
            ).hexdigest(),
            "log_sha256": None,
            "stdout_sha256": None,
            "xml_sha256": None,
        },
        "completion_sha256",
    )
    _qualification_write_new(
        qualification_root / "private-aggregate.complete.json",
        contract.canonical_json_bytes(aggregate_completion),
        code="preflight_qualification_aggregate_unwritable",
    )
    aggregate_marker_overrun = time.monotonic() >= deadline_monotonic
    if aggregate_marker_overrun and not aggregate_finalization_overrun:
        aggregate_deadline_failure = _self_hash(
            {
                "schema_version": QUALIFICATION_COMPLETION_SCHEMA_VERSION,
                "suite_identity": contract.QUALIFICATION_SUITE_ID,
                "phase": "aggregate",
                "mode": "aggregate",
                "status": "failed",
                "terminal_reason": "timeout",
                "deadline_overrun": True,
                "result_sha256": private_aggregate[
                    "private_aggregate_sha256"
                ],
                "completion_sha256": aggregate_completion[
                    "completion_sha256"
                ],
            },
            "deadline_failure_sha256",
        )
        _qualification_write_new(
            qualification_root / "private-aggregate.deadline-failure.json",
            contract.canonical_json_bytes(aggregate_deadline_failure),
            code="preflight_qualification_aggregate_unwritable",
        )
    if aggregate_finalization_overrun or aggregate_marker_overrun:
        _reject("preflight_qualification_timeout")
    _qualification_require_before_deadline(deadline_monotonic)
    public_phases: list[dict[str, Any]] = []
    for phase_result in phase_results:
        collection = phase_result["collection"]
        execution = phase_result["execution"]
        public_phases.append(
            {
                "phase": phase_result["phase"],
                "node_count": phase_result["node_count"],
                "node_list_sha256": phase_result["node_list_sha256"],
                "collection_duration_ns": collection["duration_monotonic_ns"],
                "execution_duration_ns": execution["duration_monotonic_ns"],
                "collection_exit_code": collection["exit_code"],
                "execution_exit_code": execution["exit_code"],
                "passed_count": execution["passed_count"],
                "failed_count": execution["failed_count"],
                "error_count": execution["error_count"],
                "skipped_count": execution["skipped_count"],
                "xfailed_count": execution["xfailed_count"],
                "xpassed_count": execution["xpassed_count"],
                "collection_result_sha256": collection["result_sha256"],
                "execution_result_sha256": execution["result_sha256"],
                "collection_log_sha256": collection["log_sha256"],
                "execution_log_sha256": execution["log_sha256"],
                "collection_xml_sha256": collection["xml_sha256"],
                "execution_xml_sha256": execution["xml_sha256"],
                "collection_manifest_sha256": phase_result[
                    "collection_manifest_sha256"
                ],
            }
        )
    report = _self_hash(
        {
            "schema_version": QUALIFICATION_REPORT_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "status": "passed",
            "phase_count": len(public_phases),
            "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
            "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
            "command_profile_sha256": QUALIFICATION_COMMAND_PROFILE_SHA256,
            "phases": public_phases,
            "private_aggregate_sha256": private_aggregate[
                "private_aggregate_sha256"
            ],
        },
        "qualification_sha256",
    )
    del runtime_public
    validated_report = validate_qualification_report(report)
    _qualification_require_before_deadline(deadline_monotonic)
    return validated_report


def _load_private_contact(root: Path) -> str:
    path = root / PRIVATE_CONFIG_PATH
    try:
        payload = _read_regular_file(
            path,
            maximum=MAX_PRIVATE_CONFIG_BYTES,
            code="preflight_private_contact_invalid",
        )
    except V315PreflightError:
        raise
    if not payload:
        _reject("preflight_private_contact_invalid")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
        contact = value["secrets"]["sec_user_agent"]
    except (KeyError, TypeError, ValueError, UnicodeError):
        _reject("preflight_private_contact_invalid")
    if type(contact) is not str or not contact:
        _reject("preflight_private_contact_invalid")
    return contact


def build_production_dependencies() -> PreflightDependencies:
    """Lazily bind local-only production adapters; no work runs here."""

    try:
        from .sec_gemma_lean_science_v315_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v315_runner import (
            build_preflight_request_commitments,
        )
    except Exception:
        _reject("preflight_dependency_unavailable")

    cache: dict[str, str] = {}

    def contact(root: Path) -> str:
        if "value" not in cache:
            cache["value"] = _load_private_contact(root)
        return cache["value"]

    def authenticate(root: Path) -> Any:
        return authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact(root),
        )

    def tokens(root: Path) -> tuple[bytes | str, ...]:
        return (
            contact(root),
            str(root),
            str((root / V38_PRIVATE_ROOT).resolve(strict=False)),
            str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)),
        )

    return PreflightDependencies(
        inspect_repository=inspect_pushed_implementation,
        create_runtime_authority=create_preflight_runtime_authority,
        authenticate_source=authenticate,
        build_projection=build_streaming_science_projection,
        build_request_commitments=build_preflight_request_commitments,
        effect_snapshot=lambda: contract.build_effect_budgets()[
            "zero_effect_preflight"
        ],
        run_qualification=_run_frozen_qualification_suite,
        privacy_tokens=tokens,
    )


def build_production_pushed_result_dependencies() -> PushedResultDependencies:
    """Bind read-only result replay adapters; constructing them performs no work."""

    try:
        from .sec_gemma_lean_science_v315_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v315_runner import (
            _open_generation_batch,
            _pilot_evidence,
            _open_stage_slice,
            build_attempt_authority,
            build_comparison_update,
            build_continuation_preregistration,
            build_deterministic_payload,
            build_effect_report,
            build_model_plan,
            build_preflight_request_commitments,
            build_private_terminal_material,
            build_failure_terminal_material,
            build_public_effect_report,
            build_public_pause_artifact,
            build_public_science_summary,
            build_public_terminal_artifact,
            build_semantic_payload,
            evaluate_deterministic_science,
            rebuild_runtime_evidence,
        )
        from .sec_gemma_lean_science_v315_store import AttemptStore
    except Exception:
        _reject("pushed_result_dependency_unavailable")

    def authenticate_preflight(root: Path, commit: str) -> Mapping[str, Any]:
        return _authenticate_preflight_revision(
            root, preflight_commit=commit, direct_head=False
        )

    def authenticate_source(root: Path, contact: str) -> Any:
        return authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact,
        )

    def rebuild_context(
        execution_authority: Mapping[str, Any], projection: Any
    ) -> Mapping[str, Any]:
        commitments = build_preflight_request_commitments(projection)
        plan = build_model_plan(projection)
        authority = build_attempt_authority(
            execution_authority=execution_authority,
            projection=projection,
            commitments=commitments,
            plan=plan,
        )
        return {"authority": authority, "plan": plan}

    def replay_completed(**kwargs: Any) -> Mapping[str, Any]:
        store = kwargs["store"]
        authority = kwargs["authority"]
        projection = kwargs["projection"]
        plan = kwargs["plan"]
        effect_report = kwargs["effect_report"]
        guards, aggregate, latency, guard_by_request = rebuild_runtime_evidence(
            store, plan=plan
        )
        stage_slice = _open_stage_slice(store, plan=plan)
        sealed = _open_generation_batch(store, plan=plan)
        semantic = build_semantic_payload(
            plan=plan,
            sealed_calls_by_request_sha256=sealed,
            segment_guard_by_request_sha256=guard_by_request,
            runtime_aggregate=aggregate,
            latency_receipt=latency,
        )
        deterministic = build_deterministic_payload(
            plan=plan,
            stage_slice=stage_slice,
            semantic_payload=semantic,
        )
        evaluated = evaluate_deterministic_science(
            semantic_payload=semantic,
            deterministic_payload=deterministic,
        )
        if not isinstance(evaluated, Mapping) or not isinstance(
            evaluated.get("evaluation"), Mapping
        ):
            _reject("pushed_result_deterministic_replay_failed")
        evaluation = dict(evaluated["evaluation"])
        public_summary = build_public_science_summary(evaluation)
        material = build_private_terminal_material(
            authority=authority,
            projection=projection,
            plan=plan,
            runtime_guards=guards,
            runtime_aggregate=aggregate,
            latency_receipt=latency,
            stage_slice=stage_slice,
            semantic_payload=semantic,
            deterministic_payload=deterministic,
            evaluation=evaluation,
            effect_report=effect_report,
            invocation_parent=kwargs["invocation_parent"],
            invocation_parent_kind=kwargs["invocation_parent_kind"],
        )
        if material.get("science_summary") != public_summary:
            _reject("pushed_result_science_summary_replay_failed")
        return material

    def replay_failure(**kwargs: Any) -> Mapping[str, Any]:
        snapshot = kwargs["snapshot"]
        receipt = kwargs["receipt"]
        evidence = kwargs["evidence"]
        try:
            before_terminal = replace(
                snapshot,
                journal_head_sha256=receipt.journal_head_before_terminal_sha256,
            )
        except Exception:
            _reject("pushed_result_failure_replay_failed")
        return build_failure_terminal_material(
            authority=kwargs["authority"],
            terminal_code=receipt.terminal_code,
            terminal_status=receipt.status,
            effect_report=kwargs["effect_report"],
            snapshot=before_terminal,
            invocation_parent=kwargs["invocation_parent"],
            invocation_parent_kind=kwargs["invocation_parent_kind"],
            market_values_opened=evidence["market_values_opened"],
            model_responses_opened=evidence["model_responses_opened"],
            source_authenticated=evidence["source_authenticated"],
        )

    def tokens(root: Path, contact: str) -> tuple[str, ...]:
        v38 = str((root / V38_PRIVATE_ROOT).resolve(strict=False))
        v315 = str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False))
        return (
            contact,
            str(root),
            str(root).replace("\\", "/"),
            v38,
            v38.replace("\\", "/"),
            v315,
            v315.replace("\\", "/"),
        )

    return PushedResultDependencies(
        authenticate_preflight_revision=authenticate_preflight,
        load_private_contact=load_private_contact,
        authenticate_source=authenticate_source,
        build_projection=build_streaming_science_projection,
        rebuild_attempt_context=rebuild_context,
        open_store=lambda path, authority: AttemptStore.open(
            path, authority=authority
        ),
        build_effect_report=build_effect_report,
        build_public_effect_report=build_public_effect_report,
        rebuild_pilot_evidence=lambda **kwargs: _pilot_evidence(
            kwargs["store"], plan=kwargs["plan"]
        ),
        build_public_pause_artifact=build_public_pause_artifact,
        build_continuation_preregistration=build_continuation_preregistration,
        replay_completed_terminal=replay_completed,
        replay_failure_terminal=replay_failure,
        build_public_terminal_artifact=build_public_terminal_artifact,
        build_comparison_update=build_comparison_update,
        privacy_tokens=tokens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AAPL SEC/Gemma v3.15 zero-effect preflight"
    )
    parser.add_argument("repo_root", nargs="?", default=".")
    parser.add_argument(
        "--pushed-result",
        action="store_true",
        help="run only the separate read-only pushed-result authentication gate",
    )
    arguments = parser.parse_args(argv)
    try:
        result = (
            authenticate_pushed_result(Path(arguments.repo_root))
            if arguments.pushed_result
            else run_preflight(Path(arguments.repo_root))
        )
    except V315PreflightError as exc:
        print(json.dumps({"status": "failed", "code": exc.code}, sort_keys=True))
        return 1
    output = {"status": result["status"]}
    if arguments.pushed_result:
        output["pushed_result_gate_sha256"] = result[
            "pushed_result_gate_sha256"
        ]
    else:
        output["public_artifact_sha256"] = result["public_artifact_sha256"]
    print(json.dumps(output, sort_keys=True))
    return 0


__all__ = [
    "EXECUTION_AUTHORITY_SCHEMA_VERSION",
    "QUALIFICATION_REPORT_SCHEMA_VERSION",
    "QUALIFICATION_COMMAND_PROFILE_SHA256",
    "QUALIFICATION_TIMEOUT_SECONDS",
    "PREFLIGHT_ATTEMPT_ID",
    "PREFLIGHT_SCHEMA_VERSION",
    "PUSHED_RESULT_GATE_SCHEMA_VERSION",
    "PRIVATE_MANIFEST_SCHEMA_VERSION",
    "PUBLIC_ARTIFACT_SCHEMA_VERSION",
    "PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION",
    "PreflightDependencies",
    "PublicationRecoveryDependencies",
    "PushedResultDependencies",
    "REQUEST_COMMITMENTS_SCHEMA_VERSION",
    "V315PreflightError",
    "authenticate_execution_preflight",
    "authenticate_pushed_result",
    "build_production_dependencies",
    "build_production_publication_recovery_dependencies",
    "build_production_pushed_result_dependencies",
    "inspect_pushed_implementation",
    "load_execution_authority",
    "load_private_contact",
    "load_publication_recovery_authority",
    "main",
    "run_preflight",
    "reconstruct_frozen_execution_authority",
    "validate_bridge_manifest",
    "validate_qualification_report",
    "validate_private_preflight_manifest",
    "validate_public_preflight_artifact",
    "validate_public_failed_preflight_artifact",
    "validate_repository_snapshot",
    "validate_request_commitments",
]


if __name__ == "__main__":
    raise SystemExit(main())
