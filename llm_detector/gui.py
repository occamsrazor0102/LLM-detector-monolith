"""Desktop GUI for the LLM Detection Pipeline."""

import os
import threading
from collections import Counter

from llm_detector.compat import HAS_TK
from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv

if HAS_TK:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox


class DetectorGUI:
    """Simple desktop GUI for single-text and file analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Detector Pipeline v0.61")
        self.root.geometry("1040x760")

        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')

        self._build_layout()

    def _build_layout(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        file_row = ttk.Frame(frame)
        file_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(file_row, text='Input file (CSV/XLSX):').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(side=tk.LEFT)

        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(opts, text='Prompt column').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=18).grid(row=0, column=1, sticky='w', padx=6)
        ttk.Label(opts, text='Sheet (xlsx)').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=16).grid(row=0, column=3, sticky='w', padx=6)
        ttk.Label(opts, text='Attempter filter').grid(row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=18).grid(row=0, column=5, sticky='w', padx=6)

        l3 = ttk.LabelFrame(frame, text='Continuation Analysis (DNA-GPT)')
        l3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(l3, text='Provider').grid(row=0, column=0, sticky='w', padx=6, pady=6)
        ttk.Combobox(l3, textvariable=self.provider_var, values=['anthropic', 'openai'], width=12, state='readonly').grid(row=0, column=1, sticky='w', pady=6)
        ttk.Label(l3, text='API Key (optional)').grid(row=0, column=2, sticky='w', padx=(16, 6), pady=6)
        ttk.Entry(l3, textvariable=self.api_key_var, show='*').grid(row=0, column=3, sticky='ew', padx=(0, 6), pady=6)
        l3.columnconfigure(3, weight=1)

        ttk.Label(frame, text='Single text input (optional):').pack(anchor='w')
        self.text_input = tk.Text(frame, height=10, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(4, 8))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Analyze Text', command=lambda: self._run_async(self._analyze_text)).pack(side=tk.LEFT)
        ttk.Button(actions, text='Analyze File', command=lambda: self._run_async(self._analyze_file)).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text='Clear Output', command=self._clear_output).pack(side=tk.LEFT)

        ttk.Label(frame, text='Results:').pack(anchor='w')
        self.output = tk.Text(frame, height=20, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, textvariable=self.status_var).pack(anchor='w', pady=(8, 0))

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[('Data files', '*.csv *.xlsx *.xlsm'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _clear_output(self):
        self.output.delete('1.0', tk.END)
        self.status_var.set('Ready')

    def _run_async(self, fn):
        self.status_var.set('Running...')

        def runner():
            try:
                fn()
                self.root.after(0, lambda: self.status_var.set('Done'))
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set('Error'))
                self.root.after(0, lambda: messagebox.showerror('Analysis Error', str(exc)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text):
        self.root.after(0, lambda: (self.output.insert(tk.END, text), self.output.see(tk.END)))

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Enter text to analyze.'))
            return
        result = analyze_prompt(
            text,
            run_l3=True,
            api_key=self.api_key_var.get().strip() or None,
            dna_provider=self.provider_var.get(),
        )
        self._append(self._format_result(result) + '\n')

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Choose a CSV/XLSX file to analyze.'))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            tasks = load_xlsx(path, sheet=self.sheet_var.get().strip() or None, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.csv':
            tasks = load_csv(path, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        else:
            self.root.after(0, lambda: messagebox.showerror('Unsupported file', f'Unsupported extension: {ext}'))
            return
        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo('No tasks', 'No qualifying prompts found.'))
            return

        api_key = self.api_key_var.get().strip() or None
        counts = Counter()
        for i, task in enumerate(tasks, 1):
            r = analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                run_l3=True,
                api_key=api_key,
                dna_provider=self.provider_var.get(),
            )
            counts[r['determination']] += 1
            self._append(f"[{i}/{len(tasks)}] {self._format_result(r)}\n")

        summary = (
            f"\nSummary: RED={counts.get('RED', 0)} | AMBER={counts.get('AMBER', 0)} "
            f"| YELLOW={counts.get('YELLOW', 0)} | GREEN={counts.get('GREEN', 0)}\n"
        )
        self._append(summary)

    @staticmethod
    def _format_result(result):
        return (
            f"{result.get('determination')} | conf={result.get('confidence', 0):.2f} | "
            f"words={result.get('word_count', 0)} | reason={result.get('reason', '')}"
        )


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()
