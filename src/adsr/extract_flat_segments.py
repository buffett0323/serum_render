import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def extract_flat_segments(audio_path, target_duration=0.3, amplitude_threshold=0.2, 
                         flatness_threshold=0.1, sample_rate=44100):
    """
    Extract segments from audio that have flat amplitude and contain sound.
    
    Args:
        audio_path: Path to the audio file
        target_duration: Duration of segments to extract (seconds)
        amplitude_threshold: Minimum amplitude to consider as "sound"
        flatness_threshold: Maximum allowed amplitude variation for "flatness"
        sample_rate: Sample rate of the audio
    
    Returns:
        List of tuples: (start_time, segment_audio, flatness_score)
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    
    # Calculate segment length in samples
    segment_length = int(target_duration * sample_rate)
    
    # Calculate RMS for each potential segment
    segments = []
    step_size = segment_length // 4  # Overlap segments for better coverage
    
    for start_sample in range(0, len(audio) - segment_length, step_size):
        segment = audio[start_sample:start_sample + segment_length]
        
        # Check if segment has sufficient amplitude
        rms = np.sqrt(np.mean(segment**2))
        if rms < amplitude_threshold:
            continue
        
        # Check flatness (low variance in amplitude)
        # Calculate amplitude envelope
        envelope = np.abs(segment)
        amplitude_variance = np.var(envelope)
        flatness_score = 1.0 / (1.0 + amplitude_variance)  # Higher score = more flat
        
        if flatness_score > flatness_threshold:
            start_time = start_sample / sample_rate
            segments.append((start_time, segment, flatness_score))
    
    return segments

def analyze_one_shot_directory(directory_path, output_dir=None, target_duration=0.3, 
                              amplitude_threshold=0.2, flatness_threshold=0.1):
    """
    Analyze all one-shot files in a directory and extract flat segments.
    
    Args:
        directory_path: Path to directory containing one-shot files
        output_dir: Directory to save extracted segments (optional)
        target_duration: Duration of segments to extract (seconds)
        amplitude_threshold: Minimum amplitude threshold
        flatness_threshold: Flatness threshold
    """
    # Find all WAV files
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    print(f"Found {len(wav_files)} WAV files in {directory_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Statistics
    total_segments = 0
    successful_files = 0
    results = []
    
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        print(f"\nProcessing: {filename}")
        
        try:
            # Extract flat segments
            segments = extract_flat_segments(
                wav_file, 
                target_duration=target_duration,
                amplitude_threshold=amplitude_threshold,
                flatness_threshold=flatness_threshold
            )
            
            if segments:
                successful_files += 1
                total_segments += len(segments)
                
                print(f"  Found {len(segments)} flat segments")
                
                # Save segments if output directory is specified
                if output_dir:
                    base_name = os.path.splitext(filename)[0]
                    for i, (start_time, segment, flatness_score) in enumerate(segments):
                        output_filename = f"{base_name}_flat_{i:02d}_t{start_time:.2f}s.wav"
                        output_path = os.path.join(output_dir, output_filename)
                        sf.write(output_path, segment, 44100)
                
                # Store results for analysis
                for start_time, segment, flatness_score in segments:
                    rms = np.sqrt(np.mean(segment**2))
                    results.append({
                        'filename': filename,
                        'start_time': start_time,
                        'rms': rms,
                        'flatness_score': flatness_score,
                        'segment': segment
                    })
                
                # Plot the best segment for this file
                # if segments:
                    # best_segment = max(segments, key=lambda x: x[2])  # Highest flatness score
                    # plot_segment_analysis(wav_file, best_segment, filename)
            else:
                print(f"  No suitable flat segments found")
                
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {len(wav_files)}")
    print(f"Files with flat segments: {successful_files}")
    print(f"Total flat segments extracted: {total_segments}")
    
    if results:
        # Calculate statistics
        rms_values = [r['rms'] for r in results]
        flatness_scores = [r['flatness_score'] for r in results]
        
        print(f"\nSegment Statistics:")
        print(f"Average RMS: {np.mean(rms_values):.4f}")
        print(f"Average flatness score: {np.mean(flatness_scores):.4f}")
        print(f"Best flatness score: {np.max(flatness_scores):.4f}")
        print(f"Worst flatness score: {np.min(flatness_scores):.4f}")
        
        # Save summary to file
        if output_dir:
            summary_file = os.path.join(output_dir, "extraction_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Flat Segment Extraction Summary\n")
                f.write("="*40 + "\n")
                f.write(f"Total files processed: {len(wav_files)}\n")
                f.write(f"Files with flat segments: {successful_files}\n")
                f.write(f"Total flat segments extracted: {total_segments}\n")
                f.write(f"Average RMS: {np.mean(rms_values):.4f}\n")
                f.write(f"Average flatness score: {np.mean(flatness_scores):.4f}\n")
                f.write(f"Best flatness score: {np.max(flatness_scores):.4f}\n")
                f.write(f"Worst flatness score: {np.min(flatness_scores):.4f}\n")
                f.write("\nDetailed Results:\n")
                for r in results:
                    f.write(f"{r['filename']}: t={r['start_time']:.2f}s, RMS={r['rms']:.4f}, Flatness={r['flatness_score']:.4f}\n")
            
            print(f"\nSummary saved to: {summary_file}")

def plot_segment_analysis(original_file, segment_data, filename):
    """Plot analysis of a segment compared to the original file."""
    start_time, segment, flatness_score = segment_data
    
    # Load original audio for comparison
    original_audio, sr = sf.read(original_file)
    if original_audio.ndim > 1:
        original_audio = original_audio[:, 0]
    
    # Create time axes
    original_time = np.linspace(0, len(original_audio) / sr, len(original_audio))
    segment_time = np.linspace(0, len(segment) / sr, len(segment))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original audio
    ax1.plot(original_time, original_audio, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axvspan(start_time, start_time + len(segment) / sr, alpha=0.3, color='red', label='Extracted Segment')
    ax1.set_title(f'Original Audio: {filename}')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot extracted segment
    ax2.plot(segment_time, segment, 'r-', linewidth=0.8)
    ax2.set_title(f'Extracted Flat Segment (Flatness Score: {flatness_score:.4f})')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"segment_analysis_{os.path.splitext(filename)[0]}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Analysis plot saved: {plot_filename}")


if __name__ == "__main__":
    # Configuration
    one_shot_dir = "/mnt/gestalt/home/buffett/adsr/rendered_one_shot"
    output_dir = ""#"/mnt/gestalt/home/buffett/adsr/rendered_one_shot_flat"
    
    # Parameters
    target_duration = 0.2  # seconds
    amplitude_threshold = 0.1  # RMS threshold
    flatness_threshold = 0.5  # Flatness threshold (higher = more flat)
    
    print("Flat Segment Extractor")
    print("="*50)
    print(f"Target duration: {target_duration} seconds")
    print(f"Amplitude threshold: {amplitude_threshold}")
    print(f"Flatness threshold: {flatness_threshold}")
    print(f"Input directory: {one_shot_dir}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    # Run the analysis
    analyze_one_shot_directory(
        one_shot_dir,
        output_dir=output_dir,
        target_duration=target_duration,
        amplitude_threshold=amplitude_threshold,
        flatness_threshold=flatness_threshold
    ) 