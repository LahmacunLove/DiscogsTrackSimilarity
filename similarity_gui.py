#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cover Song Similarity GUI

Graphical interface for selecting releases and tracks for cover song similarity analysis.
Provides an intuitive way to:
1. Select a reference release by ID
2. Choose a track from that release
3. Configure similarity parameters
4. Run analysis and view results

@author: ffx
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import os
import sys
from pathlib import Path
import json

# Add src directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config import load_config
except ImportError:
    def load_config():
        return {"LIBRARY_PATH": "/home/ffx/.cache/discogsLibary/discogsLib"}

from similarity_analyzer import CoverSongSimilarityAnalyzer, SimilarityAlgorithm, CamelotWheel


class CoverSongGUI:
    """GUI application for cover song similarity analysis."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cover Song Similarity Analyzer")
        self.root.geometry("900x700")
        
        # Load configuration
        self.config = load_config()
        self.library_path = self.config.get("LIBRARY_PATH", "")
        
        # Initialize analyzer
        self.analyzer = None
        self.available_releases = []
        self.selected_release = None
        self.selected_track = None
        
        self.setup_ui()
        self.load_releases()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Cover Song Similarity Analyzer", 
                               font=('TkDefaultFont', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Library path
        ttk.Label(main_frame, text="Library Path:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.library_path_var = tk.StringVar(value=self.library_path)
        library_entry = ttk.Entry(main_frame, textvariable=self.library_path_var, width=60)
        library_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=2)
        ttk.Button(main_frame, text="Browse", 
                  command=self.browse_library_path).grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Release selection
        ttk.Label(main_frame, text="Release ID:").grid(row=2, column=0, sticky=tk.W, pady=(15, 2))
        self.release_id_var = tk.StringVar()
        self.release_id_var.trace('w', self.on_release_id_change)
        release_entry = ttk.Entry(main_frame, textvariable=self.release_id_var, width=20)
        release_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=(15, 2))
        
        # Release info
        self.release_info_var = tk.StringVar(value="Enter a release ID to see available tracks")
        ttk.Label(main_frame, textvariable=self.release_info_var, 
                 foreground="gray").grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Track selection
        ttk.Label(main_frame, text="Select Track:").grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        
        # Track buttons frame
        self.tracks_frame = ttk.Frame(main_frame)
        self.tracks_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding="10")
        params_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 10))
        params_frame.columnconfigure(1, weight=1)
        
        # Algorithm selection
        ttk.Label(params_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.algorithm_var = tk.StringVar(value="Combined Features")
        algorithm_combo = ttk.Combobox(params_frame, textvariable=self.algorithm_var, 
                                     values=["Combined Features", "HPCP Cross-Correlation", 
                                            "Chroma Similarity", "Harmonic Key Matching"], 
                                     state="readonly", width=25)
        algorithm_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Pitch range
        ttk.Label(params_frame, text="Pitch Range:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pitch_range_var = tk.StringVar(value="8%")
        pitch_frame = ttk.Frame(params_frame)
        pitch_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Radiobutton(pitch_frame, text="8%", variable=self.pitch_range_var, 
                       value="8%").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(pitch_frame, text="16%", variable=self.pitch_range_var, 
                       value="16%").pack(side=tk.LEFT)
        
        # Harmonic filter
        self.harmonic_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Only harmonically compatible tracks (Camelot wheel)", 
                       variable=self.harmonic_filter_var).grid(row=2, column=0, columnspan=2, 
                                                              sticky=tk.W, pady=2)
        
        # Minimum similarity
        ttk.Label(params_frame, text="Min Similarity:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.min_similarity_var = tk.DoubleVar(value=0.7)
        similarity_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, 
                                   variable=self.min_similarity_var, orient=tk.HORIZONTAL)
        similarity_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 50), pady=2)
        self.similarity_label = ttk.Label(params_frame, text="0.70")
        self.similarity_label.grid(row=3, column=2, sticky=tk.W, pady=2)
        self.min_similarity_var.trace('w', self.update_similarity_label)
        
        # Max results
        ttk.Label(params_frame, text="Max Results:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.max_results_var = tk.IntVar(value=20)
        results_spin = ttk.Spinbox(params_frame, from_=5, to=100, width=10,
                                  textvariable=self.max_results_var)
        results_spin.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=20)
        
        self.analyze_button = ttk.Button(button_frame, text="🔍 Find Similar Tracks", 
                                        command=self.start_analysis, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="📁 Refresh Releases", 
                  command=self.load_releases).pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_button = ttk.Button(button_frame, text="💾 Export Results", 
                                       command=self.export_results, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=8, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Similar Tracks", padding="5")
        results_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(10, weight=1)
        
        # Results tree
        columns = ('Similarity', 'Artist', 'Title', 'Release', 'Track', 'Key/Camelot', 'BPM', 'Danceability', 'Algorithm')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='tree headings', height=12)
        
        # Configure columns
        self.results_tree.heading('#0', text='#')
        self.results_tree.column('#0', width=40, minwidth=40)
        
        column_widths = {
            'Similarity': 80,
            'Artist': 120,
            'Title': 150,
            'Release': 120,
            'Track': 60,
            'Key/Camelot': 80,
            'BPM': 60,
            'Danceability': 80,
            'Algorithm': 120
        }
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            width = column_widths.get(col, 100)
            self.results_tree.column(col, width=width, minwidth=60)
        
        # Scrollbars for results
        results_v_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=results_v_scroll.set, xscrollcommand=results_h_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        results_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Store results for export
        self.current_results = []
    
    def browse_library_path(self):
        """Browse for library path directory."""
        path = filedialog.askdirectory(title="Select Discogs Library Directory",
                                      initialdir=self.library_path_var.get())
        if path:
            self.library_path_var.set(path)
            self.library_path = path
            self.load_releases()
    
    def load_releases(self):
        """Load available releases from the library."""
        library_path = self.library_path_var.get()
        if not library_path or not os.path.exists(library_path):
            messagebox.showerror("Error", "Please select a valid library path")
            return
        
        try:
            self.progress_var.set("Loading releases...")
            self.progress_bar.start()
            
            # Initialize analyzer
            self.analyzer = CoverSongSimilarityAnalyzer(library_path)
            self.available_releases = self.analyzer.get_available_releases()
            
            self.progress_bar.stop()
            self.progress_var.set(f"Loaded {len(self.available_releases)} releases")
            
        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Error loading releases")
            messagebox.showerror("Error", f"Error loading releases: {e}")
    
    def on_release_id_change(self, *args):
        """Handle release ID change."""
        release_id = self.release_id_var.get().strip()
        
        # Clear previous track selection
        self.clear_track_buttons()
        self.selected_track = None
        self.analyze_button.config(state=tk.DISABLED)
        
        if not release_id:
            self.release_info_var.set("Enter a release ID to see available tracks")
            return
        
        # Find release in available releases
        matching_release = None
        for release in self.available_releases:
            if release['id'] == release_id:
                matching_release = release
                break
        
        if matching_release:
            self.selected_release = matching_release
            self.release_info_var.set(f"Release: {matching_release['title']}")
            self.show_track_buttons(matching_release['tracks'])
        else:
            self.release_info_var.set(f"Release ID '{release_id}' not found or has no analyzed tracks")
    
    def clear_track_buttons(self):
        """Clear track selection buttons."""
        for widget in self.tracks_frame.winfo_children():
            widget.destroy()
    
    def show_track_buttons(self, tracks):
        """Show track selection buttons."""
        self.clear_track_buttons()
        
        for i, track_position in enumerate(tracks):
            # Get track info
            track_info = self.analyzer.get_track_info(self.selected_release['id'], track_position)
            
            # Create button text
            button_text = f"{track_position}"
            if track_info and track_info.get('title'):
                button_text += f": {track_info['title']}"
                if track_info.get('duration'):
                    button_text += f" ({track_info['duration']})"
            
            # Create button
            btn = ttk.Button(self.tracks_frame, text=button_text,
                           command=lambda tp=track_position: self.select_track(tp))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Configure column weights
        for col in range(3):
            self.tracks_frame.columnconfigure(col, weight=1)
    
    def select_track(self, track_position):
        """Select a track for analysis."""
        self.selected_track = track_position
        
        # Update button states to show selection
        for widget in self.tracks_frame.winfo_children():
            if widget.cget('text').startswith(track_position + ":") or widget.cget('text') == track_position:
                widget.config(style='Accent.TButton')
            else:
                widget.config(style='TButton')
        
        self.analyze_button.config(state=tk.NORMAL)
        
        # Get track info for display
        track_info = self.analyzer.get_track_info(self.selected_release['id'], track_position)
        if track_info and track_info.get('title'):
            self.progress_var.set(f"Selected: {track_info['title']} [{track_position}]")
        else:
            self.progress_var.set(f"Selected track: {track_position}")
    
    def update_similarity_label(self, *args):
        """Update similarity threshold label."""
        value = self.min_similarity_var.get()
        self.similarity_label.config(text=f"{value:.2f}")
    
    def start_analysis(self):
        """Start similarity analysis in a separate thread."""
        if not self.selected_release or not self.selected_track:
            messagebox.showerror("Error", "Please select a release and track")
            return
        
        # Disable UI during analysis
        self.analyze_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        
        # Start analysis thread
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()
    
    def run_analysis(self):
        """Run the similarity analysis."""
        try:
            # Update UI
            self.root.after(0, lambda: self.progress_var.set("Analyzing tracks..."))
            self.root.after(0, self.progress_bar.start)
            
            # Get parameters
            algorithm_name = self.algorithm_var.get()
            algorithm_map = {
                "Combined Features": SimilarityAlgorithm.COMBINED_FEATURES,
                "HPCP Cross-Correlation": SimilarityAlgorithm.HPCP_CROSS_CORRELATION,
                "Chroma Similarity": SimilarityAlgorithm.CHROMA_SIMILARITY,
                "Harmonic Key Matching": SimilarityAlgorithm.HARMONIC_KEY_MATCHING
            }
            algorithm = algorithm_map.get(algorithm_name, SimilarityAlgorithm.COMBINED_FEATURES)
            
            pitch_range = float(self.pitch_range_var.get().rstrip('%'))
            min_similarity = self.min_similarity_var.get()
            max_results = self.max_results_var.get()
            harmonic_filter = self.harmonic_filter_var.get()
            
            # Run analysis
            results = self.analyzer.find_similar_tracks(
                self.selected_release['id'],
                self.selected_track,
                algorithm,
                pitch_range,
                min_similarity,
                max_results,
                harmonic_filter
            )
            
            # Store results
            self.current_results = results
            
            # Update UI with results
            self.root.after(0, lambda: self.display_results(results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
        finally:
            # Re-enable UI
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
            if self.current_results:
                self.root.after(0, lambda: self.export_button.config(state=tk.NORMAL))
    
    def display_results(self, results):
        """Display analysis results in the tree view."""
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add new results
        for i, result in enumerate(results, 1):
            # Format key/camelot display
            key_display = result.get('key', '')
            camelot = result.get('camelot', '')
            if key_display and camelot:
                key_display = f"{key_display} ({camelot})"
            elif camelot:
                key_display = camelot
            
            # Format BPM
            bpm = result.get('bpm', 0.0)
            bpm_display = f"{bpm:.0f}" if bpm > 0 else ""
            
            # Format danceability
            dance = result.get('danceability', 0.0)
            dance_display = f"{dance:.2f}" if dance > 0 else ""
            
            # Truncate long titles
            title = result['track_title'][:25] + ("..." if len(result['track_title']) > 25 else "")
            release = result['release_title'][:20] + ("..." if len(result['release_title']) > 20 else "")
            
            self.results_tree.insert('', 'end', text=str(i), values=(
                f"{result['similarity']:.3f}",
                result['track_artist'][:15] + ("..." if len(result['track_artist']) > 15 else ""),
                title,
                release,
                result['track_position'],
                key_display,
                bpm_display,
                dance_display,
                result.get('algorithm', '')[:15]
            ))
        
        self.progress_var.set(f"Found {len(results)} similar tracks")
    
    def export_results(self):
        """Export results to CSV file."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.analyzer.export_results(self.current_results, filename)
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = CoverSongGUI()
    app.run()


if __name__ == '__main__':
    main()