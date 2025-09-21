"""
HR Talent Management Dashboard - Enhanced UX Edition
Production-ready application with superior user experience
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import json
import time
from io import BytesIO
from pathlib import Path
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Management ---
@dataclass
class AppConfig:
    """Centralized configuration management"""
    page_title: str = "HR Talent Management Dashboard"
    page_icon: str = "👥"
    layout: str = "wide"
    max_file_size_mb: int = 50
    cache_ttl_seconds: int = 3600
    session_timeout_minutes: int = 30
    max_upload_files: int = 100
    data_retention_days: int = 90
    enable_ai_insights: bool = True
    enable_data_export: bool = True
    log_level: str = "INFO"
    enable_guided_tour: bool = True
    rows_per_page: int = 25
    
    # Security settings
    enable_encryption: bool = True
    require_authentication: bool = False
    allowed_file_extensions: List[str] = field(default_factory=lambda: ["xlsx", "xls", "csv"])
    
    # Performance settings
    chunk_size: int = 10000
    enable_parallel_processing: bool = True
    cache_strategy: str = "aggressive"
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        config = cls()
        for key in config.__dataclass_fields__:
            env_key = f"HR_DASHBOARD_{key.upper()}"
            if env_value := os.getenv(env_key):
                setattr(config, key, env_value)
        return config

# --- Constants and Enums ---
class PerformanceLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3

class PotentialLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3

class NineBoxCategory(Enum):
    RISK = "Low Potential / Low Performance (Risk)"
    SOLID_PERFORMER = "Low Potential / Moderate Performance (Solid Performer)"
    EFFECTIVE = "Low Potential / High Performance (Effective)"
    INCONSISTENT_PLAYER = "Moderate Potential / Low Performance (Inconsistent Player)"
    CORE_PLAYER = "Moderate Potential / Moderate Performance (Core Player)"
    HIGH_PERFORMER = "Moderate Potential / High Performance (High Performer)"
    INCONSISTENT_STAR = "High Potential / Low Performance (Inconsistent Star)"
    FUTURE_STAR = "High Potential / Moderate Performance (Future Star)"
    STAR = "High Potential / High Performance (Star)"
    UNCATEGORIZED = "Uncategorized"

class ViewMode(Enum):
    ESSENTIAL = "Essential"
    PERFORMANCE = "Performance Review"
    COMPLETE = "Complete Details"

# --- View Presets ---
VIEW_PRESETS = {
    ViewMode.ESSENTIAL: [
        'Employee ID', 'Employee Name', 'Department', 
        'Designation', 'Reporting Manager'
    ],
    ViewMode.PERFORMANCE: [
        'Employee ID', 'Employee Name', 'Department',
        'Performance', 'Potential Rating', '9-Box Category',
        'Tenure (Years)'
    ],
    ViewMode.COMPLETE: []  # All columns
}

# --- Error Handling ---
class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

# --- UX Helper Functions ---
class UXHelpers:
    """User experience enhancement utilities"""
    
    @staticmethod
    def show_tooltip(text: str, help_text: str):
        """Display text with a help tooltip"""
        return st.markdown(f'{text} <span title="{help_text}">ℹ️</span>', unsafe_allow_html=True)
    
    @staticmethod
    def create_download_link(data: bytes, filename: str, text: str) -> str:
        """Create a download link for data"""
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    
    @staticmethod
    def format_metric_delta(value: float, baseline: float = 2.0) -> Tuple[str, str]:
        """Format metric delta for display"""
        if value == 0:
            return "0%", "off"
        delta = (value - baseline) / baseline * 100
        return f"{delta:+.0f}%", "normal" if delta >= 0 else "inverse"
    
    @staticmethod
    def pluralize(count: int, singular: str, plural: str = None) -> str:
        """Pluralize text based on count"""
        if plural is None:
            plural = f"{singular}s"
        return f"{count} {singular if count == 1 else plural}"
    
    @staticmethod
    def show_progress_steps(current_step: int, total_steps: int, step_names: List[str]):
        """Display progress through multiple steps"""
        progress = current_step / total_steps
        st.progress(progress)
        st.caption(f"Step {current_step} of {total_steps}: {step_names[current_step-1]}")
    
    @staticmethod
    def format_error_message(error: Exception, context: str = "") -> str:
        """Convert technical errors to user-friendly messages"""
        error_str = str(error).lower()
        
        if "invalid literal" in error_str or "could not convert" in error_str:
            return "📋 Found text in a number field. Please check that Performance and Potential columns contain only numbers 1-3."
        elif "list index out of range" in error_str:
            return "📋 The file appears to be empty or missing data. Please check your file."
        elif "keyerror" in error_str:
            column = str(error).strip("'\"")
            return f"📋 Missing required column: {column}. Please ensure your file has all required columns."
        elif "permission" in error_str:
            return "📋 Unable to access the file. Please check that it's not open in another program."
        else:
            return f"📋 {context}: {str(error)}" if context else f"📋 An error occurred: {str(error)}"

# --- Security Module ---
class SecurityManager:
    """Handles security aspects of the application"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        return Path(filename).name
    
    @staticmethod
    def validate_file_size(file, max_size_mb: int) -> bool:
        """Validate file size"""
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        return file_size <= max_size_mb * 1024 * 1024
    
    @staticmethod
    def check_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in allowed_extensions)
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Remove potentially harmful content from dataframe"""
        suspicious_patterns = ['<script', 'javascript:', 'onclick', 'onerror', 'SELECT', 'DROP', 'INSERT']
        
        for col in df.columns:
            if df[col].dtype == 'object':
                for pattern in suspicious_patterns:
                    df[col] = df[col].astype(str).str.replace(pattern, '', case=False, regex=False)
        
        return df

# --- Data Validation ---
class DataValidator:
    """Validates and cleans employee data"""
    
    CORE_COLUMNS = [
        'Employee ID', 'Employee Name', 'Department', 
        'Performance', 'Potential Rating'
    ]
    
    RECOMMENDED_COLUMNS = [
        'Designation', 'Reporting Manager', 'Date of Joining', 'HOD'
    ]

    ALL_EXPECTED_COLUMNS = CORE_COLUMNS + RECOMMENDED_COLUMNS
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate dataframe schema against core and recommended columns"""
        missing_core_cols = [col for col in cls.CORE_COLUMNS if col not in df.columns]
        missing_recommended_cols = [col for col in cls.RECOMMENDED_COLUMNS if col not in df.columns]
        
        is_valid = len(missing_core_cols) == 0
        return is_valid, missing_core_cols, missing_recommended_cols
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types"""
        type_mappings = {
            'Employee ID': 'str',
            'Performance': 'float',
            'Potential Rating': 'float',
            'Date of Joining': 'datetime64[ns]'
        }
        
        for col, dtype in type_mappings.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dtype, errors='ignore')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        return df
    
    @staticmethod
    def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
        """Validate numeric ranges"""
        for col in ['Performance', 'Potential Rating']:
            if col in df.columns:
                df.loc[~df[col].between(1, 3), col] = np.nan
        
        if 'Tenure (Years)' in df.columns:
            df.loc[df['Tenure (Years)'] < 0, 'Tenure (Years)'] = np.nan
        
        return df

# --- Data Processing Module (REVISED - MORE FLEXIBLE) ---
# --- Data Processing Module (FINAL, ROBUST VERSION) ---
class DataProcessor:
    """Handles all data processing operations with enhanced flexibility and intelligence"""

    # --- Aliases for expected column names ---
    COLUMN_ALIASES = {
        'Employee ID': ['employee id', 'emp id', 'id', 'employee #', 'emp #'],
        'Employee Name': ['employee name', 'name', 'full name'],
        'Department': ['department', 'dept', 'business unit'],
        'Performance': ['performance', 'perf score', 'performance rating', 'perf.'],
        'Potential Rating': ['potential rating', 'potential', 'potential score'],
        'Designation': ['designation', 'job title', 'title', 'role'],
        'Reporting Manager': ['reporting manager', 'manager', 'reports to', 'supervisor'],
        'Date of Joining': ['date of joining', 'doj', 'hire date', 'start date'],
        'HOD': ['hod', 'head of department', 'department head']
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.security = SecurityManager()
        self.validator = DataValidator()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes DataFrame columns by mapping aliases to canonical names."""
        rename_map = {}
        # First, strip any whitespace from all column headers
        df.columns = df.columns.str.strip()
        
        for col in df.columns:
            cleaned_col = str(col).strip().lower().replace('_', ' ')
            for canonical_name, aliases in self.COLUMN_ALIASES.items():
                if cleaned_col in aliases:
                    rename_map[col] = canonical_name
                    break
        df.rename(columns=rename_map, inplace=True)
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_and_consolidate_data(_self, uploaded_files: List) -> Tuple[pd.DataFrame, List[str]]:
        """Load and consolidate data with robust, intelligent parsing for complex file structures."""
        if not uploaded_files:
            return pd.DataFrame(), []
        
        all_dfs = []
        warnings = []
        successful_files = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                filename = _self.security.sanitize_filename(file.name)

                # Security checks (unchanged)
                if not _self.security.check_file_extension(filename, _self.config.allowed_file_extensions):
                    warnings.append(f"⚠️ Skipped {filename}: Invalid file type.")
                    continue
                
                df = pd.DataFrame()
                file_buffer = BytesIO(file.getvalue())

                # --- NEW, ROBUST PARSING LOGIC ---
                # 1. Read the CSV/Excel without assuming a header row.
                if filename.lower().endswith('.csv'):
                    temp_df = pd.read_csv(file_buffer, header=None, encoding='utf-8', sep=None, engine='python')
                else: # For .xlsx/.xls
                    temp_df = pd.read_excel(file_buffer, header=None)

                # 2. Find the actual header row by searching for key columns.
                header_row_index = -1
                for i, row in temp_df.iterrows():
                    row_str = ' '.join(str(x).lower() for x in row.dropna())
                    if 'employee id' in row_str and 'employee name' in row_str:
                        header_row_index = i
                        break
                
                if header_row_index == -1:
                    warnings.append(f"ℹ️ Skipped {filename}: Could not find a valid header row with 'Employee ID' and 'Employee Name'.")
                    continue
                
                # 3. Re-create the DataFrame using the correct header.
                header = temp_df.iloc[header_row_index]
                df = temp_df.iloc[header_row_index + 1:].copy()
                df.columns = header
                df.reset_index(drop=True, inplace=True)

                # 4. Drop fully empty columns (often at the start or end).
                df.dropna(axis=1, how='all', inplace=True)
                df = df.loc[:, ~df.columns.isna()] # Drop columns with NaN as header name

                if df.empty:
                    warnings.append(f"⚠️ Skipped {filename}: No data found below the header row.")
                    continue

                # Standardize column names using aliases
                df = _self._standardize_columns(df)

                # --- Data Cleaning and Validation ---
                if 'Employee ID' not in df.columns:
                    warnings.append(f"ℹ️ Skipped {filename}: File does not appear to contain required data after cleaning.")
                    continue

                # Clean up rows where Employee ID is missing (removes junk rows)
                df.dropna(subset=['Employee ID'], inplace=True)
                if df.empty:
                    warnings.append(f"⚠️ Skipped {filename}: No valid employee rows found.")
                    continue

                # Convert text ratings to numeric, correctly handling 'Select'
                rating_map = {'low': 1, 'moderate': 2, 'high': 3, 'select': np.nan}
                for col in ['Performance', 'Potential Rating']:
                    if col in df.columns:
                        # Use pd.to_numeric for robustness, coercing errors to NaN
                        df[col] = df[col].astype(str).str.strip().str.lower().map(rating_map).fillna(df[col])
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Validate schema and data types
                is_valid, missing_core, missing_recommended = _self.validator.validate_schema(df)
                if not is_valid:
                    warnings.append(f"🔴 Skipped {filename}: Missing critical columns: {', '.join(missing_core)}")
                    continue
                
                # Add missing recommended columns if they don't exist
                for col in missing_recommended:
                    if col not in df.columns:
                        df[col] = None

                df = _self.validator.validate_data_types(df)
                df = _self.validator.validate_ranges(df)
                
                all_dfs.append(df)
                successful_files.append(filename)
                logger.info(f"Successfully loaded and parsed {filename} with {len(df)} records")
                
            except Exception as e:
                user_friendly_error = UXHelpers.format_error_message(e, f"Error processing {filename}")
                warnings.append(user_friendly_error)
                logger.error(f"Failed to process {filename}: {e}", exc_info=True)
        
        if not all_dfs:
            return pd.DataFrame(), warnings
        
        # Consolidate and finalize the master DataFrame
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        consolidated_df = _self._calculate_tenure(consolidated_df)
        consolidated_df = _self._add_nine_box_categories(consolidated_df)
        
        if 'Employee ID' in consolidated_df.columns:
            original_count = len(consolidated_df)
            consolidated_df = consolidated_df.drop_duplicates(subset=['Employee ID'], keep='last')
            if len(consolidated_df) < original_count:
                warnings.append(f"ℹ️ Removed {original_count - len(consolidated_df)} duplicate employee records.")
        
        success_message = f"✅ Successfully loaded {UXHelpers.pluralize(len(successful_files), 'file')} with {len(consolidated_df)} unique employees."
        warnings.insert(0, success_message)
        
        return consolidated_df, warnings
    
    # --- Unchanged Methods ---
    @staticmethod
    def _calculate_tenure(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date of Joining' in df.columns:
            df['Date of Joining'] = pd.to_datetime(df['Date of Joining'], errors='coerce')
            current_date = pd.Timestamp.now()
            df['Tenure (Years)'] = (current_date - df['Date of Joining']).dt.days / 365.25
            df['Tenure (Years)'] = df['Tenure (Years)'].round(1)
            df.loc[df['Tenure (Years)'] < 0, 'Tenure (Years)'] = 0
        return df

    @staticmethod
    def _get_nine_box_category(performance: float, potential: float) -> str:
        if pd.isna(performance) or pd.isna(potential):
            return NineBoxCategory.UNCATEGORIZED.value
        if potential >= 3:
            if performance < 2: return NineBoxCategory.INCONSISTENT_STAR.value
            elif performance < 3: return NineBoxCategory.FUTURE_STAR.value
            else: return NineBoxCategory.STAR.value
        elif potential >= 2:
            if performance < 2: return NineBoxCategory.INCONSISTENT_PLAYER.value
            elif performance < 3: return NineBoxCategory.CORE_PLAYER.value
            else: return NineBoxCategory.HIGH_PERFORMER.value
        else:
            if performance < 2: return NineBoxCategory.RISK.value
            elif performance < 3: return NineBoxCategory.SOLID_PERFORMER.value
            else: return NineBoxCategory.EFFECTIVE.value

    def _add_nine_box_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Performance' in df.columns and 'Potential Rating' in df.columns:
            df['9-Box Category'] = df.apply(
                lambda row: self._get_nine_box_category(row['Performance'], row['Potential Rating']), axis=1)
        return df

    @staticmethod
    def export_to_excel(df: pd.DataFrame) -> BytesIO:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Employee_Data')
            workbook = writer.book
            worksheet = writer.sheets['Employee_Data']
            header_format = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
                worksheet.set_column(idx, idx, min(max_length, 50))
        output.seek(0)
        return output

    @staticmethod
    def create_sample_template() -> BytesIO:
        sample_data = {
            'Employee ID': ['EMP001', 'EMP002', 'EMP003'],
            'Employee Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Department': ['Sales', 'Marketing', 'IT'],
            'Designation': ['Manager', 'Senior Executive', 'Developer'],
            'Reporting Manager': ['Alice Brown', 'Alice Brown', 'Charlie Davis'],
            'Date of Joining': ['2020-01-15', '2019-06-01', '2021-03-10'],
            'HOD': ['Alice Brown', 'Alice Brown', 'Charlie Davis'],
            'Performance': [3, 2, 2.5],
            'Potential Rating': [2.5, 3, 2]
        }
        df = pd.DataFrame(sample_data)
        return DataProcessor.export_to_excel(df)

# --- Visualization Module ---
# --- Visualization Module (REVISED TO PREVENT PLOTLY CRASH) ---
class VisualizationManager:
    """Manages all dashboard visualizations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.color_scheme = {
            'primary': '#4472C4',
            'secondary': '#ED7D31',
            'success': '#70AD47',
            'warning': '#FFC000',
            'danger': '#C5504B'
        }
    
    def create_nine_box_grid(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive 9-box grid visualization"""
        if '9-Box Category' not in df.columns:
            return None
        
        # Prepare grid data
        grid_data = []
        for perf in [1, 2, 3]:
            for pot in [1, 2, 3]:
                # Broaden the filter to handle float values (e.g., 2.5)
                employees = df[(df['Performance'] >= perf) & (df['Performance'] < perf + 1) &
                               (df['Potential Rating'] >= pot) & (df['Potential Rating'] < pot + 1)]
                count = len(employees)
                names = employees['Employee Name'].head(3).tolist()
                
                grid_data.append({
                    'Performance_Grid': perf,
                    'Potential_Grid': pot,
                    'Count': count,
                    'Names': names
                })
        
        grid_df = pd.DataFrame(grid_data)
        
        # Create hover text
        hover_text = []
        for _, row in grid_df.iterrows():
            category = DataProcessor._get_nine_box_category(row['Performance_Grid'], row['Potential_Grid'])
            employees_str = ', '.join(row['Names'])
            if len(row['Names']) == 3 and row['Count'] > 3:
                employees_str += f" (+{row['Count']-3} more)"
            
            text = f"<b>{category}</b><br>Count: {row['Count']}"
            if row['Count'] > 0:
                text += f"<br>Employees: {employees_str}"
            hover_text.append(text)
        
        pivot_data = grid_df.pivot(index='Potential_Grid', columns='Performance_Grid', values='Count')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=['Low (1-2)', 'Moderate (2-3)', 'High (3)'],
            y=['Low (1-2)', 'Moderate (2-3)', 'High (3)'],
            colorscale='Viridis',
            text=pivot_data.values,
            texttemplate='%{text}',
            textfont={"size": 16, "color": "white"},
            customdata=np.array(hover_text).reshape(3, 3),
            hovertemplate='%{customdata}<extra></extra>',
            colorbar=dict(title="Count", thickness=15)
        ))
        
        fig.update_layout(
            title={'text': '🎯 Talent 9-Box Grid', 'font': {'size': 22, 'color': '#2c3e50'}},
            xaxis=dict(title='Performance →', side='bottom', tickfont={'size': 14}),
            yaxis=dict(title='Potential →', tickfont={'size': 14}),
            height=500,
            margin=dict(l=80, r=20, t=60, b=60)
        )
        
        return fig
    
    def create_department_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create department distribution chart"""
        dept_dist = df['Department'].value_counts().head(10)
        
        fig = px.bar(
            x=dept_dist.values,
            y=dept_dist.index,
            orientation='h',
            text=dept_dist.values,
            color=dept_dist.values,
            color_continuous_scale='Blues',
            labels={'x': 'Number of Employees', 'y': 'Department'}
        )
        
        fig.update_traces(
            textposition='outside',
            texttemplate='%{text}',
            hovertemplate='<b>%{y}</b><br>Employees: %{x}<extra></extra>'
        )
        
        fig.update_layout(
            title='Employee Distribution by Department',
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        
        return fig
    
    def create_tenure_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create tenure distribution histogram"""
        if 'Tenure (Years)' not in df.columns or df['Tenure (Years)'].isnull().all():
            fig = go.Figure()
            fig.update_layout(
                title='Employee Tenure Distribution',
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[{'text': 'No "Date of Joining" data available', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16, 'color': 'grey'}}],
                height=400
            )
            return fig

        fig = px.histogram(
            df,
            x='Tenure (Years)',
            nbins=20,
            title='Employee Tenure Distribution',
            labels={'count': 'Number of Employees', 'Tenure (Years)': 'Tenure (Years)'},
            color_discrete_sequence=[self.color_scheme['primary']],
            hover_data={'Tenure (Years)': ':.1f'}
        )
        
        avg_tenure = df['Tenure (Years)'].mean()
        
        fig.add_vline(
            x=avg_tenure,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_tenure:.1f} years",
            annotation_position="top right"
        )
        
        fig.update_layout(height=400, showlegend=False, hovermode='x unified')
        
        return fig
    
    # --- METHOD WITH THE FIX ---
    def create_performance_potential_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Create performance vs potential scatter plot"""
        df_plot = df.copy()
        df_plot['Performance_jitter'] = df_plot['Performance'] + np.random.uniform(-0.08, 0.08, len(df_plot))
        df_plot['Potential_jitter'] = df_plot['Potential Rating'] + np.random.uniform(-0.08, 0.08, len(df_plot))
        
        size_param = 'Tenure (Years)' if 'Tenure (Years)' in df.columns and df['Tenure (Years)'].notna().any() else None

        # --- FIX APPLIED HERE ---
        # If tenure is used for size, fill any missing values (NaN) with 0.
        # This prevents the Plotly error when 'Date of Joining' is missing for some employees.
        if size_param:
            df_plot[size_param].fillna(0, inplace=True)
            # Also ensure no negative tenure values are passed to the size parameter.
            df_plot.loc[df_plot[size_param] < 0, size_param] = 0

        fig = px.scatter(
            df_plot,
            x='Performance_jitter',
            y='Potential_jitter',
            color='Department',
            hover_name='Employee Name',
            hover_data={
                'Performance': ':.1f',
                'Potential Rating': ':.1f',
                'Designation': True,
                'Tenure (Years)': ':.1f',
                'Performance_jitter': False,
                'Potential_jitter': False
            },
            title='📊 Performance vs. Potential Analysis',
            size=size_param,
            size_max=12,
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis=dict(title='Performance →', tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Moderate', 'High'], range=[0.5, 3.5]),
            yaxis=dict(title='Potential →', tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Moderate', 'High'], range=[0.5, 3.5]),
            height=500,
            hovermode='closest'
        )
        
        fig.add_hline(y=2, line_dash="dot", line_color="gray", opacity=0.4)
        fig.add_vline(x=2, line_dash="dot", line_color="gray", opacity=0.4)
        
        fig.add_annotation(x=1.5, y=3.3, text="🌟 High Potential", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=3.3, y=1.5, text="⚡ High Performance", showarrow=False, font=dict(color="blue"))
        
        return fig
    
    def create_category_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """Create 9-box category breakdown pie chart"""
        category_dist = df['9-Box Category'].value_counts()
        
        color_map = {
            NineBoxCategory.STAR.value: '#2ecc71',
            NineBoxCategory.FUTURE_STAR.value: '#3498db',
            NineBoxCategory.HIGH_PERFORMER.value: '#9b59b6',
            NineBoxCategory.CORE_PLAYER.value: '#f39c12',
            NineBoxCategory.EFFECTIVE.value: '#1abc9c',
            NineBoxCategory.SOLID_PERFORMER.value: '#95a5a6',
            NineBoxCategory.INCONSISTENT_STAR.value: '#e67e22',
            NineBoxCategory.INCONSISTENT_PLAYER.value: '#d35400',
            NineBoxCategory.RISK.value: '#e74c3c',
            NineBoxCategory.UNCATEGORIZED.value: '#bdc3c7'
        }
        
        colors = [color_map.get(cat, '#95a5a6') for cat in category_dist.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=category_dist.index,
            values=category_dist.values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Talent Category Distribution',
            height=450,
            annotations=[dict(text='Talent<br>Mix', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig


# --- Analytics Module ---
class AnalyticsEngine:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key HR metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_employees'] = len(df)
        metrics['departments'] = df['Department'].nunique()
        metrics['average_tenure'] = df['Tenure (Years)'].mean() if 'Tenure (Years)' in df.columns else 0
        
        # Performance metrics
        if 'Performance' in df.columns:
            metrics['avg_performance'] = df['Performance'].mean()
            metrics['high_performers'] = len(df[df['Performance'] >= 2.5])
            metrics['low_performers'] = len(df[df['Performance'] < 2])
        
        # Potential metrics
        if 'Potential Rating' in df.columns:
            metrics['avg_potential'] = df['Potential Rating'].mean()
            metrics['high_potential'] = len(df[df['Potential Rating'] >= 2.5])
        
        # 9-Box distribution
        if '9-Box Category' in df.columns:
            category_counts = df['9-Box Category'].value_counts().to_dict()
            metrics['stars'] = category_counts.get(NineBoxCategory.STAR.value, 0)
            metrics['future_stars'] = category_counts.get(NineBoxCategory.FUTURE_STAR.value, 0)
            metrics['risk_employees'] = category_counts.get(NineBoxCategory.RISK.value, 0)
        
        # Turnover risk
        metrics['turnover_risk'] = 0
        if all(col in df.columns for col in ['Performance', 'Tenure (Years)']) and df['Tenure (Years)'].notna().any():
            metrics['turnover_risk'] = len(df[(df['Performance'] < 2) | (df['Tenure (Years)'] < 1)])
        
        return metrics
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, metrics: Dict) -> Dict[str, List[str]]:
        """Generate categorized data-driven insights"""
        insights = {
            'success': [],
            'warning': [],
            'info': [],
            'danger': []
        }
        
        # Performance insights
        if 'avg_performance' in metrics:
            avg_perf = metrics['avg_performance']
            if avg_perf >= 2.5:
                insights['success'].append(f"Strong overall performance ({avg_perf:.2f}/3.0)")
            elif avg_perf >= 2:
                insights['info'].append(f"Moderate performance level ({avg_perf:.2f}/3.0)")
            else:
                insights['warning'].append(f"Performance needs improvement ({avg_perf:.2f}/3.0)")
        
        # Talent pipeline
        if 'future_stars' in metrics and 'stars' in metrics:
            total_top = metrics['stars'] + metrics['future_stars']
            pct = (total_top / metrics['total_employees']) * 100 if metrics['total_employees'] > 0 else 0
            
            if pct >= 20:
                insights['success'].append(f"Robust talent pipeline: {pct:.0f}% are top talent")
            elif pct >= 10:
                insights['info'].append(f"Moderate talent pipeline: {pct:.0f}% are top talent")
            else:
                insights['warning'].append(f"Talent pipeline needs development: only {pct:.0f}% are top talent")
        
        # Retention risk
        if 'turnover_risk' in metrics and ('Tenure (Years)' in df.columns and df['Tenure (Years)'].notna().any()):
            risk_pct = (metrics['turnover_risk'] / metrics['total_employees']) * 100 if metrics['total_employees'] > 0 else 0
            if risk_pct > 20:
                insights['danger'].append(f"High turnover risk: {risk_pct:.0f}% show warning signs")
            elif risk_pct > 10:
                insights['warning'].append(f"Moderate turnover risk: {risk_pct:.0f}% need attention")
            else:
                insights['success'].append(f"Low turnover risk: only {risk_pct:.0f}% at risk")
        
        # Department performance
        if 'Department' in df.columns and 'Performance' in df.columns:
            dept_perf = df.groupby('Department')['Performance'].mean()
            if len(dept_perf) > 1:
                top_dept = dept_perf.idxmax()
                bottom_dept = dept_perf.idxmin()
                insights['info'].append(f"{top_dept} leads in performance, {bottom_dept} needs support")
        
        return insights

# --- AI Integration Module ---
class AIInsightsGenerator:
    """AI-powered insights using Google Gemini"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.initialized = False
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.initialized = False
        else:
            self.initialized = False
    
    def generate_strategic_insights(self, df: pd.DataFrame, metrics: Dict) -> Optional[str]:
        """Generate AI-powered strategic insights"""
        if not self.model:
            return None
        
        try:
            summary = self._prepare_data_summary(df, metrics)
            
            prompt = f"""
            As an expert HR strategist, analyze this talent data and provide actionable insights.
            
            **Data Summary:**
            {summary}
            
            Provide a concise analysis with:
            1. Three key observations
            2. Two critical risks
            3. Three specific action items
            
            Format as markdown with clear sections.
            Keep response under 400 words.
            Focus on practical, implementable recommendations.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI insight generation failed: {e}")
            return None
    
    def _prepare_data_summary(self, df: pd.DataFrame, metrics: Dict) -> str:
        """Prepare concise data summary for AI"""
        summary_parts = [
            f"Total Employees: {metrics.get('total_employees', 'N/A')}",
            f"Average Performance: {metrics.get('avg_performance', 'N/A'):.2f}" if 'avg_performance' in metrics else "Performance data not available",
            f"Average Potential: {metrics.get('avg_potential', 'N/A'):.2f}" if 'avg_potential' in metrics else "Potential data not available",
            f"Stars: {metrics.get('stars', 0)}",
            f"Future Stars: {metrics.get('future_stars', 0)}",
            f"Risk Employees: {metrics.get('risk_employees', 0)}",
            f"Average Tenure: {metrics.get('average_tenure', 'N/A'):.1f} years" if 'average_tenure' in metrics else "Tenure data not available"
        ]
        
        if 'Department' in df.columns:
            dept_summary = df['Department'].value_counts().head(5).to_dict()
            summary_parts.append(f"Top Departments: {dept_summary}")
        
        return "\n".join(summary_parts)

# --- Session State Manager ---
class SessionStateManager:
    """Manages application session state"""
    
    @staticmethod
    def init_session_state():
        """Initialize session state variables"""
        defaults = {
            'authenticated': False,
            'data_loaded': False,
            'last_activity': datetime.now(),
            'export_history': [],
            'tour_completed': False,
            'filter_state': {},
            'previous_filters': {},
            'page_number': 0,
            'view_mode': ViewMode.ESSENTIAL,
            'show_welcome': True,
            'insights_cache': None,
            'last_upload_time': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def check_session_timeout(timeout_minutes: int = 30):
        """Check and handle session timeout"""
        if datetime.now() - st.session_state.last_activity > timedelta(minutes=timeout_minutes):
            st.session_state.clear()
            st.warning("⏱️ Your session has expired for security. Please refresh the page to start again.")
            st.stop()
        
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def save_filter_state():
        """Save current filter state for undo functionality"""
        st.session_state.previous_filters = st.session_state.filter_state.copy()
    
    @staticmethod
    def restore_filter_state():
        """Restore previous filter state"""
        if st.session_state.previous_filters:
            st.session_state.filter_state = st.session_state.previous_filters.copy()
            return True
        return False

# --- Advanced PDF Reporting Module ---
import base64
from weasyprint import HTML, CSS
from datetime import datetime

class ReportGenerator:
    """Handles the creation of beautiful, story-like PDF reports from dashboard data."""

    @staticmethod
    def create_story_report(df: pd.DataFrame, metrics: dict, viz_manager: VisualizationManager, company_name: str) -> bytes:
        """
        Generates a professionally styled PDF report using HTML and CSS.
        """
        # --- 1. Generate charts and encode them for HTML ---
        charts = {
            'nine_box': viz_manager.create_nine_box_grid(df),
            'scatter': viz_manager.create_performance_potential_scatter(df),
            'category': viz_manager.create_category_breakdown(df),
            'department': viz_manager.create_department_distribution(df),
        }
        encoded_charts = {}
        for name, fig in charts.items():
            if fig:
                img_bytes = fig.to_image(format="png", width=800, height=450, scale=2)
                encoded_string = base64.b64encode(img_bytes).decode()
                encoded_charts[name] = f"data:image/png;base64,{encoded_string}"

        # --- 2. Build the HTML content for the report ---
        report_date = datetime.now().strftime("%d %B %Y")

        # Calculate key percentages for the executive summary
        total_employees = metrics.get('total_employees', 1)
        top_talent_count = metrics.get('stars', 0) + metrics.get('future_stars', 0)
        top_talent_pct = (top_talent_count / total_employees * 100) if total_employees > 0 else 0
        risk_count = metrics.get('risk_employees', 0)
        risk_pct = (risk_count / total_employees * 100) if total_employees > 0 else 0

        html_content = f"""
        <html>
            <head>
                <style>
                    /* --- Define the styles (CSS) --- */
                    @page {{ size: A4; margin: 1.5cm; }}
                    body {{ font-family: 'Helvetica', sans-serif; color: #333; }}
                    h1 {{ color: #153d6d; font-size: 28px; text-align: center; margin-bottom: 0; }}
                    h2 {{ color: #153d6d; font-size: 22px; border-bottom: 2px solid #4472C4; padding-bottom: 5px; margin-top: 30px;}}
                    h3 {{ color: #4472C4; font-size: 16px; margin-bottom: 10px; }}
                    p, li {{ font-size: 12px; line-height: 1.6; }}
                    .cover-page {{ text-align: center; margin-top: 150px; }}
                    .subtitle {{ font-size: 18px; color: #555; text-align: center; }}
                    .report-meta {{ font-size: 14px; color: #777; text-align: center; margin-top: 50px; }}
                    .page-break {{ page-break-before: always; }}
                    .metric-container {{ display: flex; justify-content: space-around; text-align: center; margin: 20px 0; }}
                    .metric {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; width: 150px;}}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #ED7D31; }}
                    .metric-label {{ font-size: 12px; color: #555; margin-top: 5px;}}
                    .chart {{ width: 100%; margin-top: 20px; }}
                    .insight {{ background-color: #f0f5fa; border-left: 4px solid #4472C4; padding: 15px; margin: 15px 0; }}
                </style>
            </head>
            <body>
                <div class="cover-page">
                    <h1>Talent Management Report</h1>
                    <p class="subtitle">{company_name}</p>
                    <div style="height: 100px;"></div>

                    <p class="report-meta">Generated on: {report_date}</p>
                </div>

                <div class="page-break"></div>

                <h2>Executive Summary</h2>
                <p>
                    This report provides a comprehensive overview of the talent landscape at {company_name}.
                    Based on the analysis of <strong>{total_employees} employees</strong>, we have identified key strengths, potential risks, and strategic opportunities for talent development.
                </p>
                <div class="insight">
                    <p>
                        The average performance rating stands at <strong>{metrics.get('avg_performance', 0):.2f}/3.0</strong>, indicating a solid foundation. 
                        A key highlight is that <strong>{top_talent_pct:.0f}%</strong> of the workforce is identified as 'Top Talent' (Stars and Future Stars), representing a strong leadership pipeline.
                        Conversely, <strong>{risk_pct:.0f}%</strong> of employees fall into the 'At Risk' category, requiring immediate attention.
                    </p>
                </div>

                <h3>Key Performance Indicators</h3>
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{metrics.get('total_employees', 0):,}</div>
                        <div class="metric-label">Total Employees</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{metrics.get('avg_performance', 0):.2f}</div>
                        <div class="metric-label">Avg. Performance</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{top_talent_count:,}</div>
                        <div class="metric-label">Top Talent</div>
                    </div>
                     <div class="metric">
                        <div class="metric-value">{risk_count:,}</div>
                        <div class="metric-label">At Risk</div>
                    </div>
                </div>

                <div class="page-break"></div>
                <h2>Deep Dive: Talent Distribution</h2>
                <h3>Talent 9-Box Grid</h3>
                <p>The 9-Box Grid below visualizes the distribution of talent based on current performance and future potential. The concentration in each box indicates the health of our talent pipeline.</p>
                <img src="{encoded_charts['nine_box']}" class="chart">

                <h3>Talent Category Breakdown</h3>
                <p>This donut chart shows the percentage of employees in each talent category. 'Core Players' and 'High Performers' form the backbone of the organization.</p>
                <img src="{encoded_charts['category']}" class="chart">

                <div class="page-break"></div>
                <h2>Deep Dive: Performance Analysis</h2>
                <h3>Performance vs. Potential Scatter Plot</h3>
                <p>This scatter plot provides a granular view of every employee, colour-coded by department. It helps identify outliers and departmental trends in performance and potential.</p>
                <img src="{encoded_charts['scatter']}" class="chart">

                <div class="page-break"></div>
                <h2>Conclusion & Recommendations</h2>
                <p>The analysis reveals a capable and promising workforce, with a significant pool of high-potential individuals. To capitalize on these strengths and mitigate risks, we recommend the following actions:</p>

                <div class="insight">
                    <h3>1. Accelerate Top Talent Development</h3>
                    <p>Focus on the <strong>{top_talent_count} Stars and Future Stars</strong> with fast-track leadership programs and mentorship opportunities to ensure they are engaged and prepared for future roles.</p>
                </div>
                <div class="insight">
                    <h3>2. Address At-Risk Employees</h3>
                    <p>Implement targeted Performance Improvement Plans (PIPs) for the <strong>{risk_count} employees</strong> in the 'Risk' category. Proactive intervention is crucial to improve performance or manage transitions.</p>
                </div>
                 <div class="insight">
                    <h3>3. Empower Core Players</h3>
                    <p>Recognize and develop the large group of 'Core Players'. They are vital to the organization's stability and success. Provide skill enhancement opportunities to keep them motivated and effective.</p>
                </div>
		
            </body>
        </html>
        """
	
        # --- 3. Convert HTML to PDF ---
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
# --- Main Application Class ---
class HRDashboardApp:
    """Main application class with enhanced UX"""
    
    def __init__(self):
        self.config = AppConfig.from_env()
        self.processor = DataProcessor(self.config)
        self.viz = VisualizationManager(self.config)
        self.analytics = AnalyticsEngine()
        self.ai_generator = None
        self.reporter = ReportGenerator()
        
        # Initialize session state
        SessionStateManager.init_session_state()
    
    def run(self):
        """Run the application"""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout=self.config.layout,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://docs.streamlit.io',
                'Report a bug': None,
                'About': "# HR Talent Management Dashboard\nVersion 2.0\nEnterprise-grade talent analytics"
            }
        )
        
        SessionStateManager.check_session_timeout(self.config.session_timeout_minutes)
        self._apply_custom_css()
        
        if st.session_state.show_welcome and not st.session_state.tour_completed:
            self._show_guided_tour()
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title(f"{self.config.page_icon} HR Talent Management Dashboard")
            if st.session_state.data_loaded:
                last_update = st.session_state.get('last_upload_time', datetime.now())
                st.caption(f"Last updated: {last_update.strftime('%B %d, %Y at %I:%M %p')}")
        
        with st.sidebar:
            self._render_sidebar()
        
        if st.session_state.get('data_loaded'):
            self._render_main_content()
        else:
            self._render_welcome_screen()
        
        self._render_footer()

    # --- SIDEBAR METHOD WITH THE FIX ---
    def _render_sidebar(self):
        """Render sidebar with progressive disclosure and better filter controls"""
        st.markdown("## 🎯 Control Panel")
        
        with st.expander("📁 **Step 1: Upload Data**", expanded=not st.session_state.data_loaded):
            uploaded_files = st.file_uploader(
                "Select Excel or CSV files",
                type=self.config.allowed_file_extensions,
                accept_multiple_files=True,
                help=f"Upload up to {self.config.max_upload_files} files (max {self.config.max_file_size_mb}MB each)",
                key="file_uploader"
            )
            
            if uploaded_files:
                # When new files are uploaded, reset the filter state to ensure 'All Departments' is default
                if 'uploaded_files_list' not in st.session_state or st.session_state.uploaded_files_list != uploaded_files:
                    st.session_state.filter_state = {}
                    st.session_state.uploaded_files_list = uploaded_files

                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    master_df, warnings = self.processor.load_and_consolidate_data(uploaded_files)
                    
                    for warning in warnings:
                        if warning.startswith("✅"): st.success(warning)
                        elif warning.startswith("⚠️"): st.warning(warning)
                        elif warning.startswith("ℹ️"): st.info(warning)
                        else: st.error(warning)
                    
                    if not master_df.empty:
                        st.session_state.master_df = master_df
                        st.session_state.data_loaded = True
                        st.session_state.last_upload_time = datetime.now()
                        st.balloons()
            
            st.markdown("---")
            st.markdown("**Need a template?**")
            template = self.processor.create_sample_template()
            st.download_button(
                label="📄 Download Sample Template",
                data=template,
                file_name="hr_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        
        if st.session_state.get('data_loaded'):
            with st.expander("🔍 **Step 2: Filter Data**", expanded=True):
                df = st.session_state.master_df
                departments = sorted(df['Department'].dropna().unique())

                # --- FIX APPLIED HERE: "Select All" is now the default and easy to access ---
                st.markdown("**Department Filter:**")
                
                # Add "Select All" and "Clear" buttons for convenience
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All", use_container_width=True):
                        st.session_state.filter_state['departments'] = departments
                        st.rerun()
                with col2:
                    if st.button("Clear All", use_container_width=True):
                        st.session_state.filter_state['departments'] = []
                        st.rerun()

                # Set default to all departments if not already set in the session state
                if 'departments' not in st.session_state.filter_state:
                    st.session_state.filter_state['departments'] = departments
                
                selected_departments = st.multiselect(
                    "Select Departments",
                    departments,
                    default=st.session_state.filter_state.get('departments', []),
                    label_visibility="collapsed"
                )
                st.session_state.filter_state['departments'] = selected_departments
                
                # Performance and Potential sliders
                perf_range = st.slider("Performance Range", 1.0, 3.0, st.session_state.filter_state.get('performance_range', (1.0, 3.0)), 0.1)
                st.session_state.filter_state['performance_range'] = perf_range
                
                potential_range = st.slider("Potential Range", 1.0, 3.0, st.session_state.filter_state.get('potential_range', (1.0, 3.0)), 0.1)
                st.session_state.filter_state['potential_range'] = potential_range
                
                # Apply filters
                filtered_df = df[
                    (df['Department'].isin(selected_departments)) &
                    (df['Performance'].between(perf_range[0], perf_range[1])) &
                    (df['Potential Rating'].between(potential_range[0], potential_range[1]))
                ]
                st.session_state.filtered_df = filtered_df
                
                # Filter summary
                total, filtered = len(df), len(filtered_df)
                percentage = (filtered / total * 100) if total > 0 else 0
                st.metric("Filtered Results", f"{filtered:,} employees", f"{percentage:.0f}% of total", delta_color="off")
                
                # Reset button
                if st.button("↺ Reset All Filters", use_container_width=True):
                    st.session_state.filter_state = {}
                    st.rerun()

            with st.expander("💾 **Step 3: Export Data**", expanded=False):
                export_choice = st.radio("What to export?", ["Current View (Filtered)", "All Data"], key="export_choice")
                export_df = filtered_df if export_choice == "Current View (Filtered)" else df
                
                st.info(f"📊 Will export {len(export_df):,} records")
                
                if st.button("📥 Generate Excel Export", key="export_btn"):
                    excel_file = self.processor.export_to_excel(export_df)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="💾 Download Excel File",
                        data=excel_file,
                        file_name=f"hr_data_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    st.success("✅ Export ready for download!")
                st.markdown("---")
                # Replace the old PDF button with the new one
                if st.button("📄 Generate Story Report (PDF)", key="story_pdf_btn"):
                    with st.spinner("🎨 Crafting your beautiful report..."):
                        df_to_export = st.session_state.get('filtered_df', st.session_state.master_df)
                        metrics = self.analytics.calculate_metrics(df_to_export)
                        # Call the new story report generator
                        pdf_file = self.reporter.create_story_report(
                            df_to_export, 
                            metrics, 
                            self.viz,
                            company_name="Your Company Name" # You can make this dynamic
                        )
                        timestamp = datetime.now().strftime('%Y%m%d')
                        st.download_button(
                            label="📥 Download Story Report",
                            data=pdf_file,
                            file_name=f"HR_Story_Report_{timestamp}.pdf",
                            mime="application/pdf",
                        )
                    st.success("✅ Your report is ready for download!")

    # (Make sure you keep the other methods like run, _apply_custom_css, _render_main_content, etc., as they are)

    def _apply_custom_css(self):
        """Apply custom CSS for better UX"""
        st.markdown("""
        <style>
        /* Main layout improvements */
        .main {
            padding-top: 1rem;
        }
        
        /* Better tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f0f2f6;
            padding: 5px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 5px;
            color: #4a5568;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Enhanced metrics styling */
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Better file uploader */
        .uploadedFile {
            background-color: #f7fafc;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            padding: 8px;
            margin: 4px 0;
        }
        
        /* Improved buttons */
        .stButton > button {
            background-color: #4472C4;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            border: none;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #3056A0;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Success/Warning/Error styling */
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 12px;
            border-radius: 6px;
            color: #155724;
            margin: 10px 0;
        }
        
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 12px;
            border-radius: 6px;
            color: #856404;
            margin: 10px 0;
        }
        
        /* Sidebar improvements */
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        
        /* Data table improvements */
        .dataframe {
            font-size: 14px;
        }
        
        .dataframe thead tr th {
            background-color: #4472C4 !important;
            color: white !important;
            font-weight: 600;
            text-align: left;
            padding: 10px;
        }
        
        .dataframe tbody tr:hover {
            background-color: #f0f2f6 !important;
        }
        
        /* Help text styling */
        .help-text {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        /* Progress bar enhancement */
        .stProgress > div > div {
            background-color: #4472C4;
        }
        
        /* Expander improvements */
        .streamlit-expanderHeader {
            background-color: #f7fafc;
            border-radius: 6px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _show_guided_tour(self):
        """Show an interactive guided tour for first-time users"""
        with st.container():
            tour = st.expander("🎯 **Welcome! New to the Dashboard?**", expanded=True)
            with tour:
                st.markdown("""
                ### Quick Start Guide
                
                Follow these simple steps to get started:
                
                1️⃣ **Upload Your Data**
                    - Click on 'Browse files' in the sidebar
                    - Select one or more Excel or CSV files with employee data
                
                2️⃣ **Explore Your Insights**
                    - View key metrics at the top
                    - Navigate through tabs for different analyses
                    - Use filters to focus on specific groups
                
                3️⃣ **Export & Share**
                    - Download filtered data anytime
                    - Generate AI insights (optional)
                
                💡 **Pro Tips:**
                - Use the 9-Box Grid to identify top talent
                - Check the Overview tab for quick insights
                """)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col2:
                    if st.button("✅ Got it, don't show again"):
                        st.session_state.tour_completed = True
                        st.session_state.show_welcome = False
                        st.rerun()
    
    def _render_welcome_screen(self):
        """Render an engaging welcome screen"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #2c3e50; font-size: 3rem;'>👥 Welcome to Your HR Dashboard</h1>
            <p style='color: #7f8c8d; font-size: 1.2rem;'>Transform your employee data into actionable talent insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class='help-text'><h3>📊 Smart Analytics</h3><p>Automated performance and potential analysis with interactive visualizations</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class='help-text'><h3>🎯 9-Box Talent Grid</h3><p>Instantly categorize employees and identify your stars and future leaders</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class='help-text'><h3>🤖 AI Insights</h3><p>Get strategic recommendations powered by advanced AI analysis</p></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👈 **Ready to begin? Upload your data using the sidebar to get started!**")
    
    def _render_main_content(self):
        """Render main dashboard content with enhanced UX"""
        df = st.session_state.get('filtered_df', st.session_state.master_df)
        
        if df.empty:
            st.warning("No data matches your current filters. Try adjusting the filter criteria.")
            return
        
        metrics = self.analytics.calculate_metrics(df)
        
        st.markdown("### 📈 Key Performance Indicators")
        self._render_enhanced_metrics(metrics)
        st.markdown("---")
        
        tabs = st.tabs(["📊 Overview", "🎯 9-Box Grid", "📈 Analytics", "👥 Employee Data", "🤖 AI Insights"])
        
        with tabs[0]: self._render_overview_tab(df, metrics)
        with tabs[1]: self._render_nine_box_tab(df)
        with tabs[2]: self._render_analytics_tab(df, metrics)
        with tabs[3]: self._render_data_tab(df)
        with tabs[4]: self._render_ai_insights_tab(df, metrics)

    def _render_enhanced_metrics(self, metrics: Dict):
        """Render metrics with better visual design"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Employees", f"{metrics.get('total_employees', 0):,}", help="Total number of employees in the filtered dataset")
        
        with col2:
            avg_perf = metrics.get('avg_performance', 0)
            delta_pct, delta_color = UXHelpers.format_metric_delta(avg_perf)
            st.metric("Avg Performance", f"{avg_perf:.2f}/3.0", delta_pct, delta_color=delta_color, help="Average performance score (1-3 scale)")
        
        with col3:
            avg_potential = metrics.get('avg_potential', 0)
            delta_pct, delta_color = UXHelpers.format_metric_delta(avg_potential)
            st.metric("Avg Potential", f"{avg_potential:.2f}/3.0", delta_pct, delta_color=delta_color, help="Average potential rating (1-3 scale)")
        
        with col4:
            top_talent = metrics.get('stars', 0) + metrics.get('future_stars', 0)
            st.metric("Top Talent", f"{top_talent:,}", f"{(top_talent/metrics.get('total_employees', 1)*100):.0f}% of total", help="Stars and Future Stars combined")
        
        with col5:
            risk_count = metrics.get('risk_employees', 0)
            st.metric("At Risk", f"{risk_count:,}", f"{(risk_count/metrics.get('total_employees', 1)*100):.0f}% of total", delta_color="inverse" if risk_count > 0 else "off", help="Employees in the risk category")

    def _render_overview_tab(self, df: pd.DataFrame, metrics: Dict):
        """Render overview tab with insights"""
        insights = self.analytics.generate_insights(df, metrics)
        
        if any(insights.values()):
            st.markdown("### 💡 Key Insights")
            col1, col2 = st.columns(2)
            with col1:
                for insight in insights.get('success', []): st.success(f"✅ {insight}")
                for insight in insights.get('info', []): st.info(f"ℹ️ {insight}")
            with col2:
                for insight in insights.get('warning', []): st.warning(f"⚠️ {insight}")
                for insight in insights.get('danger', []): st.error(f"🔴 {insight}")
        
        st.markdown("---")
        st.markdown("### 📊 Visual Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = self.viz.create_department_distribution(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = self.viz.create_tenure_distribution(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig = self.viz.create_category_breakdown(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("#### 🏆 Department Performance")
            if 'Department' in df.columns and 'Performance' in df.columns:
                dept_metrics = df.groupby('Department').agg({
                    'Performance': 'mean', 'Potential Rating': 'mean', 'Employee ID': 'count'
                }).round(2)
                dept_metrics.columns = ['Avg Performance', 'Avg Potential', 'Employees']
                dept_metrics = dept_metrics.sort_values('Avg Performance', ascending=False)
                st.dataframe(
                    dept_metrics.style.background_gradient(cmap='RdYlGn', subset=['Avg Performance', 'Avg Potential']).format({'Employees': '{:,.0f}'}),
                    use_container_width=True, height=300
                )

    def _render_nine_box_tab(self, df: pd.DataFrame):
        """Render 9-box grid analysis with enhanced UX"""
        st.markdown("### 🎯 Talent 9-Box Grid Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = self.viz.create_nine_box_grid(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create 9-box grid. Please check data quality.")
        
        with col2:
            st.markdown("#### 📊 Distribution Summary")
            if '9-Box Category' in df.columns:
                category_counts = df['9-Box Category'].value_counts()
                for category in [NineBoxCategory.STAR, NineBoxCategory.FUTURE_STAR, NineBoxCategory.HIGH_PERFORMER, NineBoxCategory.RISK]:
                    count = category_counts.get(category.value, 0)
                    percentage = (count / len(df) * 100) if len(df) > 0 else 0
                    if category == NineBoxCategory.STAR: st.success(f"🌟 Stars: {count} ({percentage:.0f}%)")
                    elif category == NineBoxCategory.FUTURE_STAR: st.info(f"🚀 Future Stars: {count} ({percentage:.0f}%)")
                    elif category == NineBoxCategory.HIGH_PERFORMER: st.info(f"⚡ High Performers: {count} ({percentage:.0f}%)")
                    elif category == NineBoxCategory.RISK and count > 0: st.error(f"⚠️ At Risk: {count} ({percentage:.0f}%)")
        
        st.markdown("---")
        st.markdown("### 🎬 Recommended Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            stars_count = df[df['9-Box Category'] == NineBoxCategory.STAR.value]['Employee ID'].count() if '9-Box Category' in df.columns else 0
            st.markdown(f"<div class='success-box'><h4>🌟 Stars ({stars_count})</h4><ul><li>Fast-track for leadership roles</li><li>Assign stretch projects</li><li>Retention focus</li></ul></div>", unsafe_allow_html=True)
        with col2:
            future_stars_count = df[df['9-Box Category'] == NineBoxCategory.FUTURE_STAR.value]['Employee ID'].count() if '9-Box Category' in df.columns else 0
            st.markdown(f"<div class='warning-box'><h4>🚀 Future Stars ({future_stars_count})</h4><ul><li>Targeted development programs</li><li>Mentorship opportunities</li><li>Performance coaching</li></ul></div>", unsafe_allow_html=True)
        with col3:
            risk_count = df[df['9-Box Category'] == NineBoxCategory.RISK.value]['Employee ID'].count() if '9-Box Category' in df.columns else 0
            if risk_count > 0:
                st.markdown(f"<div class='warning-box' style='border-color: #f5c6cb; background-color: #f8d7da; color: #721c24;'><h4>⚠️ At Risk ({risk_count})</h4><ul><li>Performance improvement plans</li><li>Role reassignment evaluation</li><li>Exit planning if needed</li></ul></div>", unsafe_allow_html=True)

    def _render_analytics_tab(self, df: pd.DataFrame, metrics: Dict):
        """Render analytics tab with enhanced visualizations"""
        st.markdown("### 📈 Performance & Potential Analytics")
        fig = self.viz.create_performance_potential_scatter(df)
        st.plotly_chart(fig, use_container_width=True)

    def _render_data_tab(self, df: pd.DataFrame):
        """Render employee data tab with enhanced UX"""
        st.markdown("### 👥 Employee Data Explorer")
        
        search_term = st.text_input("🔍 Search", placeholder="Name, ID, or department...", help="Search across all fields")
        
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = df[mask]
            st.info(f"🔍 Found {len(display_df)} matching records")
        else:
            display_df = df
        
        st.dataframe(display_df, use_container_width=True)
        
        if st.button("📥 Export This View"):
            excel_file = self.processor.export_to_excel(display_df)
            st.download_button(
                label="💾 Download Excel",
                data=excel_file,
                file_name=f"employee_data_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    def _render_ai_insights_tab(self, df: pd.DataFrame, metrics: Dict):
        """Render AI insights tab with improved UX"""
        st.markdown("### 🤖 AI-Powered Strategic Insights")
        
        if not self.config.enable_ai_insights:
            st.warning("AI insights are currently disabled.")
            return
        
        if st.session_state.insights_cache:
            st.markdown("#### 📝 Previous Analysis")
            st.markdown(st.session_state.insights_cache['insights'])
            st.caption(f"Generated at {st.session_state.insights_cache['timestamp'].strftime('%I:%M %p')}")
            if st.button("🔄 Generate New Insights"):
                st.session_state.insights_cache = None
                st.rerun()
            return
        
        with st.expander("🔐 API Configuration", expanded=True):
            st.info("This feature uses Google Gemini AI to provide strategic recommendations. Your data is not stored by the AI service.")
            api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Enter your API key here...")
            st.markdown("<small>Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)</small>", unsafe_allow_html=True)
        
        if api_key:
            if st.button("✨ Generate AI Insights", use_container_width=True):
                with st.spinner("🧠 Analyzing your talent data..."):
                    try:
                        self.ai_generator = AIInsightsGenerator(api_key)
                        insights = self.ai_generator.generate_strategic_insights(df, metrics)
                        if insights:
                            st.success("✅ AI analysis complete!")
                            st.markdown("### 📋 Strategic Recommendations")
                            st.markdown(insights)
                            st.session_state.insights_cache = {'timestamp': datetime.now(), 'insights': insights}
                        else:
                            st.error("Unable to generate insights. Please verify your API key.")
                    except Exception as e:
                        st.error(f"An error occurred: {UXHelpers.format_error_message(e)}")

    def _render_footer(self):
        """Render footer with helpful information"""
        st.markdown("---")
        st.caption("Version 2.0 | © 2025 HR Analytics by Karthik Krishnan")

# --- Application Entry Point ---
def main():
    """Main entry point for the application"""
    try:
        app = HRDashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {UXHelpers.format_error_message(e)}")
        st.info("Please refresh the page to try again. If the problem persists, contact support.")

if __name__ == "__main__":
    main()





