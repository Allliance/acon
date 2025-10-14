"""
Report Generator module for Blueprint task evaluation results.

This module provides classes and functions to generate reports in various formats
from evaluation results data.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional


class ReportGenerator:
    """Generates evaluation reports in different formats."""
    
    # Section titles for app categories
    SECTION_TITLES = ["Single App", "Two Apps", "Three Apps"]
    
    @staticmethod
    def format_overall_report(stats: Dict[str, Any]) -> str:
        """
        Format the overall evaluation report as a string without error stats.
        
        Args:
            stats: Dictionary containing evaluation statistics
            
        Returns:
            Formatted report string with overall statistics
        """
        overall_report = []
        overall_report.append(f"{stats['model_name']} {stats['tag_name']}")
        
        # Format app tag stats
        # New app tag stats schema:
        # stats['app_tags'][num_app_tag] = {
        #        'success_avg': success_avg,
        #        'success_std': success_std,
        #        'success_total': success_total,
        #        'result_available_avg': result_available_avg,
        #        'result_available_std': result_available_std,
        #        'result_available_total': result_available_total,
        #    }

        for num_app_tag in stats['app_tags']:
            success = stats['app_tags'][num_app_tag]['success_avg']
            success_std = stats['app_tags'][num_app_tag]['success_std']
            total_available = stats['app_tags'][num_app_tag]['result_available_avg']
            total = stats['app_tags'][num_app_tag]['result_available_total']
            
            # Add step metrics if available
            steps_info = ""
            if 'steps_avg' in stats['app_tags'][num_app_tag]:
                steps_avg = stats['app_tags'][num_app_tag]['steps_avg']
                steps_std = stats['app_tags'][num_app_tag]['steps_std']
                #steps_total = stats['app_tags'][num_app_tag]['steps_total']
                steps_info = f", Steps: {steps_avg:.1f}±{steps_std:.2f}" # (total={steps_total})"

            overall_report.append(f"Num_App_Tag {num_app_tag}: {success:.1f}±{success_std:.2f}/{int(total_available)}={success/max(total_available,1)*100:.3f}% (of {int(total)}){steps_info}")

        #overall_report.append("\n")
        # Format overall stats
        overall_stats = stats['overall']
        success_rate = overall_stats['success_avg']/max(overall_stats['result_available_avg'],1)*100
        
        # Add overall step metrics if available
        steps_info = ""
        if 'steps_avg' in overall_stats:
            steps_avg = overall_stats['steps_avg']
            steps_std = overall_stats['steps_std']
            #steps_total = overall_stats['steps_total']
            steps_info = f"\nSteps: {steps_avg:.1f}±{steps_std:.2f}"

        overall_report.append(f"Overall: {overall_stats['success_avg']:.1f}/{int(overall_stats['result_available_avg'])}={success_rate:.3f}% (of {int(overall_stats['result_available_total'])}){steps_info}")

        # Format tool stats
        overall_report.append("====================================")
        overall_report.append("Tool Stats:")
        for tool, count in stats['tool_stats'].items():
            overall_report.append(f"{tool}: {count}")
        
        return "\n".join(overall_report)
    
    @staticmethod
    def format_error_report(stats: Dict[str, Any]) -> str:
        """
        Format the error report as a string.
        
        Args:
            stats: Dictionary containing evaluation statistics
            
        Returns:
            Formatted error report string
        """
        error_report = []
        error_report.append("====================================")
        error_report.append("Error Stats:")
        
        for error, count in stats['error_stats'].items():
            error_report.append(f"{error}: {count}")
        
        error_report.append("====================================")
        
        return "\n".join(error_report)
    
    @staticmethod
    def make_html_div(task_id: str, subtask_id: str, is_success: List[bool], result_available: List[bool], apps: List[str]=None, model_name: str = None, tag_name: str = None) -> str:
        """
        Create an HTML div for a task result.
        
        Args:
            task_id: Task identifier
            subtask_id: Subtask identifier
            is_success: List of booleans indicating whether trials were successful
            result_available: List of booleans indicating whether results are available for trials
            model_name: Model name to include in the query string (optional)
            tag_name: Tag name to include in the query string (optional)
            
        Returns:
            HTML div string
        """
        try:
            task_text = json.load(open(f"./tasks/{task_id}/subtasks/{subtask_id}.json"))['task']
            
            # Calculate the overall success rate
            total_available = sum(result_available)
            successful = sum(is_success)
            success_rate = successful / total_available if total_available > 0 else 0
            apps_str = "Apps: " + ", ".join(apps) if apps else ""
            
            hover_text = f"{task_id}.{subtask_id}: {task_text}\nSuccess: {successful}/{total_available}={success_rate:.3f}\n{apps_str}"
            font_color = "white"
            # Determine color based on success rate
            if total_available == 0:
                color = "gray"
            elif success_rate >= 0.75:  # At least 75% of the trials succeeded
                color = "green"
            elif success_rate >= 0.25:  # At least 25% of the trials succeeded
                color = "yellow"
                font_color = "black"
            else:  # Less than 25% of the trials succeeded
                color = "red"
            
            # Build the query string with model and tag if provided
            query_params = []
            if model_name:
                query_params.append(f"model={model_name}")
            if tag_name:
                query_params.append(f"tag={tag_name}")
            
            query_string = f"?{'&'.join(query_params)}" if query_params else ""
            
            # Add task-div class and data-search attribute for searchability
            return f'<a href="/task/{task_id}.{subtask_id}{query_string}"><div class="task-div" data-search="{hover_text}" style="background-color: {color}; color: {font_color}; padding: 10px; margin: 5px; border-radius: 5px;" title="{hover_text}">{task_id}.{subtask_id}</div></a>'
        except Exception as e:
            logging.error(f"Error creating HTML div for {task_id}.{subtask_id}: {e}")
            return f'<div class="task-div" style="background-color: gray; color: white; padding: 10px; margin: 5px; border-radius: 5px;">{task_id}.{subtask_id} (Error)</div>'
    
    @staticmethod
    def make_html_report(stats: Dict[str, Any], html_results: Dict[str, List[str]], result_path: Optional[str]) -> str:
        """
        Generate an HTML report from evaluation results.
        
        Args:
            stats: Dictionary containing evaluation statistics
            html_results: Dictionary mapping app tags to HTML divs
            result_path: Path to save the HTML report (optional)
            
        Returns:
            HTML report string
        """
        # Generate report lines using the helper methods
        overall_report = ReportGenerator.format_overall_report(stats)
        error_report = ReportGenerator.format_error_report(stats)
        
        html_content = """
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .container { display: flex; flex-wrap: wrap; }
                .task { margin: 10px; }
                .error-report { margin-top: 30px; }
                .search-box {
                    margin: 20px 0;
                    width: 50%;
                    padding: 10px;
                    font-size: 16px;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                }
                .search-box:focus {
                    outline: none;
                    border-color: #4d90fe;
                }
                .hidden {
                    display: none !important;
                }
                .section-container {
                    margin-bottom: 20px;
                }
                .section-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .section-header h2 {
                    margin-right: 10px;
                    margin-bottom: 0;
                }
                .task-count {
                    background-color: #eee;
                    padding: 5px 10px;
                    border-radius: 10px;
                    font-size: 14px;
                    color: #333;
                }
            </style>
            <script>
                // Reload the page every 60 seconds
                setTimeout(function() {
                    window.location.reload();
                }, 60000);
                
                // Filter tasks based on search input
                function filterTasks() {
                    const searchText = document.getElementById('search-box').value.toLowerCase();
                    const taskDivs = document.querySelectorAll('.task-div');
                    let visibleCount = {};
                    
                    taskDivs.forEach(div => {
                        const searchData = div.getAttribute('data-search').toLowerCase();
                        const container = div.closest('.section-container');
                        const sectionId = container ? container.getAttribute('id') : null;
                        
                        if (searchText === '' || searchData.includes(searchText)) {
                            div.parentElement.classList.remove('hidden');
                            if (sectionId) {
                                visibleCount[sectionId] = (visibleCount[sectionId] || 0) + 1;
                            }
                        } else {
                            div.parentElement.classList.add('hidden');
                        }
                    });
                    
                    // Update task counts for each section
                    Object.keys(visibleCount).forEach(sectionId => {
                        const countElement = document.querySelector(`#${sectionId} .task-count`);
                        if (countElement) {
                            const totalCount = parseInt(countElement.getAttribute('data-total'));
                            countElement.textContent = `Showing ${visibleCount[sectionId]} of ${totalCount}`;
                        }
                    });
                }
                
                // Initialize counts when the page loads
                document.addEventListener('DOMContentLoaded', () => {
                    const sections = document.querySelectorAll('.section-container');
                    sections.forEach(section => {
                        const taskDivs = section.querySelectorAll('.task-div');
                        const countElement = section.querySelector('.task-count');
                        if (countElement) {
                            countElement.setAttribute('data-total', taskDivs.length);
                            countElement.textContent = `Showing ${taskDivs.length} of ${taskDivs.length}`;
                        }
                    });
                });
            </script>
        </head>
        <body>
            <h1>Evaluation Report</h1>
        """
        html_content += "<pre>" + overall_report + "</pre>\n"
        
        # Add the search box after the overall report
        html_content += """
        <div>
            <input type="text" id="search-box" class="search-box" placeholder="Search tasks..." oninput="filterTasks()">
        </div>
        """

        for num_app_tag in html_results:
            section_id = f"section-{num_app_tag}"
            html_content += f'<div id="{section_id}" class="section-container">\n'
            html_content += f'<div class="section-header">\n'
            html_content += f'<h2>{ReportGenerator.SECTION_TITLES[int(num_app_tag)-1]}</h2>\n'
            html_content += f'<span class="task-count">Showing {len(html_results[num_app_tag])} of {len(html_results[num_app_tag])}</span>\n'
            html_content += '</div>\n'
            html_content += "<div class='container'>\n"  # Wrap the content in a div for proper spacing
            for div in html_results[num_app_tag]:
                html_content += div + "\n"
            html_content += "</div>\n"  # Close the container div
            html_content += "</div>\n"  # Close the section div
        
        # Add error report after app tag sections
        html_content += "<div class='error-report'>\n"
        html_content += "<pre>" + error_report + "</pre>\n"
        html_content += "</div>\n"
                
        html_content += """
        </body>
        </html>
        """
        
        if result_path is not None:
            try:
                with open(result_path.replace('.jsonl', '.html'), 'w') as f:
                    f.write(html_content)
                logging.debug(f"HTML report saved to {result_path.replace('.jsonl', '.html')}")
            except Exception as e:
                logging.error(f"Error writing HTML report to {result_path}: {e}")
        
        return html_content
    
    @staticmethod
    def save_json_report(stats: Dict[str, Any], result_path: str) -> None:
        """
        Save evaluation stats as a JSON file.
        
        Args:
            stats: Dictionary containing evaluation statistics
            result_path: Path to save the JSON report
        """
        try:
            json_path = result_path.replace('.jsonl', '_overall.json')
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=4)
            logging.debug(f"Overall report saved to {json_path}")
        except Exception as e:
            logging.error(f"Error writing JSON report to {result_path}: {e}")