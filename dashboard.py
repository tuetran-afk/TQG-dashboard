import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# CẤU HÌNH TRANG
# ===========================
st.set_page_config(
    page_title="Dashboard Phân tích Kết quả Thầu 2025", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD DATA
# ===========================
@st.cache_data
def load_data():
    """Đọc và xử lý dữ liệu từ file CSV"""
    try:
        # Đọc file với các tham số phù hợp
        df = pd.read_csv(
            'bckq2025.csv',  # Sửa lại tên file cho đúng
            encoding='utf-8-sig', 
            sep=';', 
            skiprows=1
        )
        
        # Làm sạch dữ liệu số
        numeric_cols = [
            'Tổng SL được phân bổ', 
            'Tổng giá trị được phân bổ', 
            'Tổng SL đã cung cấp', 
            'Tổng Giá trị cung cấp', 
            'Giá trúng thầu'
        ]
        
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                # Loại bỏ dấu phân cách
                df[col] = (df[col].astype(str)
                          .str.replace('.', '', regex=False)
                          .str.replace(',', '', regex=False))
                # Chuyển sang số
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Tính tỷ lệ thực hiện
        df['Ty_le_thuc_hien'] = np.where(
            df['Tổng giá trị được phân bổ'] > 0,
            (df['Tổng Giá trị cung cấp'] / df['Tổng giá trị được phân bổ']) * 100,
            0
        )
        
        return df
    
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file 'bckq2025.csv'. Vui lòng kiểm tra lại tên file!")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Kiểm tra dữ liệu có rỗng không
if df.empty:
    st.stop()

# ===========================
# SIDEBAR FILTERS
# ===========================
st.sidebar.image("https://via.placeholder.com/300x100.png?text=Logo+Công+Ty", use_container_width=True)
st.sidebar.header("🔍 Bộ lọc dữ liệu")

# Filter 1: Miền
if 'Miền' in df.columns:
    all_regions = df['Miền'].dropna().unique().tolist()
    region = st.sidebar.multiselect(
        "Chọn Miền:",
        options=all_regions,
        default=all_regions,
        help="Chọn một hoặc nhiều miền để phân tích"
    )
else:
    region = []
    st.sidebar.warning("⚠️ Không tìm thấy cột 'Miền'")

# Filter 2: Công ty
if 'Công ty trúng thầu' in df.columns:
    all_companies = df['Công ty trúng thầu'].dropna().unique().tolist()
    company = st.sidebar.multiselect(
        "Chọn Công ty:",
        options=all_companies,
        default=[],
        help="Để trống để xem tất cả công ty"
    )
else:
    company = []
    st.sidebar.warning("⚠️ Không tìm thấy cột 'Công ty trúng thầu'")

# Filter 3: Tỉnh
if 'Tỉnh' in df.columns:
    all_provinces = df['Tỉnh'].dropna().unique().tolist()
    province = st.sidebar.multiselect(
        "Chọn Tỉnh/Thành phố:",
        options=all_provinces,
        default=[],
        help="Để trống để xem tất cả tỉnh"
    )
else:
    province = []

# Filter 4: Hoạt chất
if 'Tên Hoạt chất' in df.columns:
    all_molecules = sorted(df['Tên Hoạt chất'].dropna().unique().tolist())
    
    # Tùy chọn tìm kiếm
    search_molecule = st.sidebar.text_input(
        "🔍 Tìm hoạt chất:",
        placeholder="Nhập tên hoạt chất...",
        help="Gõ để lọc danh sách"
    )
    
    # Lọc danh sách hoạt chất theo từ khóa tìm kiếm
    if search_molecule:
        filtered_molecules = [m for m in all_molecules if search_molecule.lower() in m.lower()]
    else:
        filtered_molecules = all_molecules
    
    molecule = st.sidebar.multiselect(
        "Chọn Hoạt chất:",
        options=filtered_molecules,
        default=[],
        help="Để trống để xem tất cả hoạt chất"
    )
    
    if molecule:
        st.sidebar.success(f"✅ Đã chọn {len(molecule)} hoạt chất")
else:
    molecule = []
    st.sidebar.warning("⚠️ Không tìm thấy cột 'Tên Hoạt chất'")

# Filter 5: Ngưỡng giá trị
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ngưỡng giá trị")
min_value = st.sidebar.number_input(
    "Giá trị phân bổ tối thiểu (VNĐ):",
    min_value=0,
    value=0,
    step=1000000,
    help="Chỉ hiển thị các bản ghi có giá trị >= ngưỡng này"
)

# ===========================
# LỌC DỮ LIỆU
# ===========================
df_filtered = df.copy()

# Áp dụng filter Miền
if region and 'Miền' in df.columns:
    df_filtered = df_filtered[df_filtered['Miền'].isin(region)]

# Áp dụng filter Công ty
if company and 'Công ty trúng thầu' in df.columns:
    df_filtered = df_filtered[df_filtered['Công ty trúng thầu'].isin(company)]

# Áp dụng filter Tỉnh
if province and 'Tỉnh' in df.columns:
    df_filtered = df_filtered[df_filtered['Tỉnh'].isin(province)]

# Áp dụng filter Hoạt chất
if molecule and 'Tên Hoạt chất' in df.columns:
    df_filtered = df_filtered[df_filtered['Tên Hoạt chất'].isin(molecule)]

# Áp dụng filter ngưỡng giá trị
if min_value > 0:
    df_filtered = df_filtered[df_filtered['Tổng giá trị được phân bổ'] >= min_value]

# Hiển thị số bản ghi sau khi lọc
st.sidebar.markdown("---")
st.sidebar.info(f"📋 **Số bản ghi hiển thị:** {len(df_filtered):,} / {len(df):,}")

# Hiển thị các bộ lọc đang áp dụng
if region or company or province or molecule or min_value > 0:
    st.sidebar.markdown("### 🔍 Bộ lọc đang áp dụng:")
    if region:
        st.sidebar.write(f"- **Miền:** {', '.join(region)}")
    if company:
        st.sidebar.write(f"- **Công ty:** {len(company)} công ty")
    if province:
        st.sidebar.write(f"- **Tỉnh:** {', '.join(province)}")
    if molecule:
        st.sidebar.write(f"- **Hoạt chất:** {len(molecule)} loại")
    if min_value > 0:
        st.sidebar.write(f"- **Giá trị tối thiểu:** {min_value:,.0f} VNĐ")

# ===========================
# MAIN DASHBOARD
# ===========================
st.markdown('<div class="main-header">📊 Dashboard Phân tích ĐTTTQG 2025</div>', unsafe_allow_html=True)
st.markdown("---")

# ===========================
# KPI METRICS
# ===========================
total_allocated = df_filtered['Tổng giá trị được phân bổ'].sum()
total_supplied = df_filtered['Tổng Giá trị cung cấp'].sum()
execution_rate = (total_supplied / total_allocated * 100) if total_allocated > 0 else 0
gap = total_allocated - total_supplied

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 Tổng giá trị phân bổ",
        value=f"{total_allocated:,.0f} VNĐ",
        delta=None
    )

with col2:
    st.metric(
        label="✅ Tổng giá trị cung cấp",
        value=f"{total_supplied:,.0f} VNĐ",
        delta=None
    )

with col3:
    # Tính tỷ lệ thực hiện của toàn bộ dữ liệu (không lọc) để so sánh
    total_allocated_all = df['Tổng giá trị được phân bổ'].sum()
    total_supplied_all = df['Tổng Giá trị cung cấp'].sum()
    execution_rate_all = (total_supplied_all / total_allocated_all * 100) if total_allocated_all > 0 else 0
    
    # Tính chênh lệch so với trung bình toàn quốc
    delta_vs_avg = execution_rate - execution_rate_all
    
    st.metric(
        label="📈 Tỷ lệ thực hiện",
        value=f"{execution_rate:.2f}%",
        delta=f"{delta_vs_avg:.2f}% so với TB toàn quốc ({execution_rate_all:.2f}%)" if len(df_filtered) < len(df) else None,
        help="Tỷ lệ % giá trị đã cung cấp / giá trị được phân bổ"
    )

with col4:
    st.metric(
        label="⚠️ Chênh lệch (chưa cung cấp)",
        value=f"{gap:,.0f} VNĐ",
        delta=None
    )

st.markdown("---")

# ===========================
# BIỂU ĐỒ HÀNG 1
# ===========================
st.subheader("📊 Phân tích Công ty & Phân bổ theo Miền")
c1, c2 = st.columns(2)

with c1:
    if 'Công ty trúng thầu' in df_filtered.columns:
        # Top 10 Công ty
        top_co = (df_filtered.groupby('Công ty trúng thầu')['Tổng giá trị được phân bổ']
                  .sum()
                  .nlargest(10)
                  .reset_index())
        
        fig_co = px.bar(
            top_co,
            x='Tổng giá trị được phân bổ',
            y='Công ty trúng thầu',
            title="🏆 Top 10 Công ty theo Giá trị Phân bổ",
            orientation='h',
            color='Tổng giá trị được phân bổ',
            color_continuous_scale='Blues',
            labels={'Tổng giá trị được phân bổ': 'Giá trị (VNĐ)'}
        )
        fig_co.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_co, use_container_width=True)
    else:
        st.warning("⚠️ Không có dữ liệu Công ty")

with c2:
    if 'Miền' in df_filtered.columns:
        # Phân bổ theo miền
        region_dist = (df_filtered.groupby('Miền')['Tổng giá trị được phân bổ']
                       .sum()
                       .reset_index())
        
        fig_region = px.pie(
            region_dist,
            values='Tổng giá trị được phân bổ',
            names='Miền',
            title="🗺️ Cơ cấu giá trị theo Miền",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_region.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Giá trị: %{value:,.0f} VNĐ<br>Tỷ lệ: %{percent}'
        )
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.warning("⚠️ Không có dữ liệu Miền")

st.markdown("---")

# ===========================
# BIỂU ĐỒ HÀNG 2
# ===========================
st.subheader("💊 Phân tích Hoạt chất & Tiến độ theo Tỉnh")
c3, c4 = st.columns(2)

with c3:
    if 'Tên Hoạt chất' in df_filtered.columns:
        # Top 10 hoạt chất
        top_ing = (df_filtered.groupby('Tên Hoạt chất')['Tổng giá trị được phân bổ']
                   .sum()
                   .nlargest(10)
                   .reset_index())
        
        fig_ing = px.bar(
            top_ing,
            x='Tên Hoạt chất',
            y='Tổng giá trị được phân bổ',
            title="💊 Top 10 Hoạt chất có giá trị thầu cao nhất",
            color='Tổng giá trị được phân bổ',
            color_continuous_scale='Greens',
            labels={'Tổng giá trị được phân bổ': 'Giá trị (VNĐ)'}
        )
        fig_ing.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig_ing, use_container_width=True)
    else:
        st.warning("⚠️ Không có dữ liệu Hoạt chất")

with c4:
    if 'Tỉnh' in df_filtered.columns:
        # Tiến độ theo tỉnh
        prov_exec = df_filtered.groupby('Tỉnh').agg({
            'Tổng giá trị được phân bổ': 'sum',
            'Tổng Giá trị cung cấp': 'sum'
        }).reset_index()
        
        prov_exec['% Thực hiện'] = np.where(
            prov_exec['Tổng giá trị được phân bổ'] > 0,
            (prov_exec['Tổng Giá trị cung cấp'] / prov_exec['Tổng giá trị được phân bổ'] * 100),
            0
        )
        
        top_prov = prov_exec.nlargest(10, 'Tổng giá trị được phân bổ')
        
        fig_prov = px.bar(
            top_prov,
            x='Tỉnh',
            y='% Thực hiện',
            title="📍 Tỷ lệ thực hiện tại 10 Tỉnh có giá trị phân bổ lớn nhất",
            color='% Thực hiện',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            labels={'% Thực hiện': 'Tỷ lệ (%)'}
        )
        fig_prov.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        # Thêm đường trung bình toàn quốc
        avg_execution_rate = (df_filtered['Tổng Giá trị cung cấp'].sum() / 
                              df_filtered['Tổng giá trị được phân bổ'].sum() * 100)
        
        fig_prov.add_hline(
            y=avg_execution_rate, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Trung bình: {avg_execution_rate:.1f}%",
            annotation_position="right"
        )
        st.plotly_chart(fig_prov, use_container_width=True)
    else:
        st.warning("⚠️ Không có dữ liệu Tỉnh")

st.markdown("---")

# ===========================
# PHÂN TÍCH CHI TIẾT HOẠT CHẤT (nếu được chọn)
# ===========================
if molecule and 'Tên Hoạt chất' in df_filtered.columns:
    st.subheader(f"💊 Phân tích chi tiết {len(molecule)} Hoạt chất đã chọn")
    
    # Tạo bảng tổng hợp cho các hoạt chất đã chọn
    molecule_summary = df_filtered.groupby('Tên Hoạt chất').agg({
        'Tổng giá trị được phân bổ': 'sum',
        'Tổng Giá trị cung cấp': 'sum',
        'Tổng SL được phân bổ': 'sum',
        'Tổng SL đã cung cấp': 'sum'
    }).reset_index()
    
    molecule_summary['Tỷ lệ thực hiện (%)'] = np.where(
        molecule_summary['Tổng giá trị được phân bổ'] > 0,
        (molecule_summary['Tổng Giá trị cung cấp'] / molecule_summary['Tổng giá trị được phân bổ'] * 100),
        0
    )
    
    molecule_summary['Chênh lệch giá trị'] = (
        molecule_summary['Tổng giá trị được phân bổ'] - 
        molecule_summary['Tổng Giá trị cung cấp']
    )
    
    # Sắp xếp theo tỷ lệ thực hiện
    molecule_summary = molecule_summary.sort_values('Tỷ lệ thực hiện (%)', ascending=False)
    
    # Hiển thị bảng tổng hợp
    col_mol1, col_mol2 = st.columns([2, 1])
    
    with col_mol1:
        st.markdown("#### 📊 Bảng tổng hợp")
        # Kiểm tra số cells
        mol_cells = len(molecule_summary) * len(molecule_summary.columns)
        if mol_cells < 100000:
            # Tạo hàm highlight dựa trên tỷ lệ
            def highlight_rate(val):
                if pd.isna(val):
                    return ''
                try:
                    num_val = float(val)
                    if num_val >= 40:
                        return 'background-color: #90EE90'  # Xanh lá nhạt
                    elif num_val >= 20:
                        return 'background-color: #FFD700'  # Vàng
                    else:
                        return 'background-color: #FFB6C1'  # Đỏ nhạt
                except:
                    return ''
            
            st.dataframe(
                molecule_summary.style.format({
                    'Tổng giá trị được phân bổ': '{:,.0f}',
                    'Tổng Giá trị cung cấp': '{:,.0f}',
                    'Tổng SL được phân bổ': '{:,.0f}',
                    'Tổng SL đã cung cấp': '{:,.0f}',
                    'Tỷ lệ thực hiện (%)': '{:.2f}',
                    'Chênh lệch giá trị': '{:,.0f}'
                }).applymap(highlight_rate, subset=['Tỷ lệ thực hiện (%)']),
                use_container_width=True,
                height=300
            )
        else:
            st.dataframe(molecule_summary, use_container_width=True, height=300)
    
    with col_mol2:
        st.markdown("#### 🎯 Thống kê nhanh")
        avg_exec = molecule_summary['Tỷ lệ thực hiện (%)'].mean()
        max_exec = molecule_summary['Tỷ lệ thực hiện (%)'].max()
        min_exec = molecule_summary['Tỷ lệ thực hiện (%)'].min()
        
        st.metric("Trung bình", f"{avg_exec:.1f}%")
        st.metric("Cao nhất", f"{max_exec:.1f}%")
        st.metric("Thấp nhất", f"{min_exec:.1f}%")
        
        # Phân loại
        good = len(molecule_summary[molecule_summary['Tỷ lệ thực hiện (%)'] >= 40])
        medium = len(molecule_summary[(molecule_summary['Tỷ lệ thực hiện (%)'] >= 20) & 
                                      (molecule_summary['Tỷ lệ thực hiện (%)'] < 40)])
        poor = len(molecule_summary[molecule_summary['Tỷ lệ thực hiện (%)'] < 20])
        
        st.markdown("**Phân loại:**")
        st.write(f"- 🟢 Tốt (≥40%): {good}")
        st.write(f"- 🟡 TB (20-40%): {medium}")
        st.write(f"- 🔴 Kém (<20%): {poor}")
    
    # Biểu đồ so sánh
    st.markdown("#### 📈 So sánh các hoạt chất")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Biểu đồ tỷ lệ thực hiện
        fig_mol_exec = px.bar(
            molecule_summary,
            x='Tỷ lệ thực hiện (%)',
            y='Tên Hoạt chất',
            orientation='h',
            title="Tỷ lệ thực hiện theo hoạt chất",
            color='Tỷ lệ thực hiện (%)',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        fig_mol_exec.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_mol_exec, use_container_width=True)
    
    with col_chart2:
        # Biểu đồ giá trị
        fig_mol_value = go.Figure()
        fig_mol_value.add_trace(go.Bar(
            y=molecule_summary['Tên Hoạt chất'],
            x=molecule_summary['Tổng giá trị được phân bổ'],
            name='Phân bổ',
            orientation='h',
            marker_color='lightblue'
        ))
        fig_mol_value.add_trace(go.Bar(
            y=molecule_summary['Tên Hoạt chất'],
            x=molecule_summary['Tổng Giá trị cung cấp'],
            name='Đã cung cấp',
            orientation='h',
            marker_color='darkblue'
        ))
        fig_mol_value.update_layout(
            title="Giá trị phân bổ vs Đã cung cấp",
            barmode='group',
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_mol_value, use_container_width=True)
    
    # Phân tích theo công ty (cho hoạt chất đã chọn)
    if 'Công ty trúng thầu' in df_filtered.columns and len(molecule_summary) <= 20:
        st.markdown("#### 🏢 Công ty cung cấp các hoạt chất này")
        
        company_mol = df_filtered.groupby(['Công ty trúng thầu', 'Tên Hoạt chất']).agg({
            'Tổng giá trị được phân bổ': 'sum',
            'Tổng Giá trị cung cấp': 'sum'
        }).reset_index()
        
        company_mol['Tỷ lệ (%)'] = np.where(
            company_mol['Tổng giá trị được phân bổ'] > 0,
            (company_mol['Tổng Giá trị cung cấp'] / company_mol['Tổng giá trị được phân bổ'] * 100),
            0
        )
        
        # Pivot để tạo heatmap
        pivot_data = company_mol.pivot_table(
            index='Công ty trúng thầu',
            columns='Tên Hoạt chất',
            values='Tỷ lệ (%)',
            fill_value=0
        )
        
        # Chỉ lấy top 15 công ty có tổng giá trị lớn nhất
        top_companies = (df_filtered.groupby('Công ty trúng thầu')['Tổng giá trị được phân bổ']
                        .sum()
                        .nlargest(15)
                        .index)
        
        pivot_data_top = pivot_data.loc[pivot_data.index.isin(top_companies)]
        
        if not pivot_data_top.empty:
            fig_heatmap = px.imshow(
                pivot_data_top,
                labels=dict(x="Hoạt chất", y="Công ty", color="Tỷ lệ (%)"),
                title="Heatmap: Tỷ lệ thực hiện theo Công ty × Hoạt chất",
                color_continuous_scale='RdYlGn',
                aspect='auto',
                range_color=[0, 100]
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.caption("💡 Màu xanh = Cung ứng tốt | Màu đỏ = Cung ứng kém")

st.markdown("---")

# ===========================
# BIỂU ĐỒ BỔ SUNG
# ===========================
st.subheader("📈 Phân tích chi tiết")

tab1, tab2, tab3 = st.tabs(["🏥 Theo CSYT", "📊 Theo Nhóm", "📉 Tồn kho cảnh báo"])

with tab1:
    if 'Tên CSYT' in df_filtered.columns:
        csyt_stats = df_filtered.groupby('Tên CSYT').agg({
            'Tổng giá trị được phân bổ': 'sum',
            'Tổng Giá trị cung cấp': 'sum'
        }).reset_index()
        
        csyt_stats['% Thực hiện'] = np.where(
            csyt_stats['Tổng giá trị được phân bổ'] > 0,
            (csyt_stats['Tổng Giá trị cung cấp'] / csyt_stats['Tổng giá trị được phân bổ'] * 100),
            0
        )
        
        top_csyt = csyt_stats.nlargest(15, 'Tổng giá trị được phân bổ')
        
        fig_csyt = go.Figure()
        fig_csyt.add_trace(go.Bar(
            x=top_csyt['Tên CSYT'],
            y=top_csyt['Tổng giá trị được phân bổ'],
            name='Phân bổ',
            marker_color='lightblue'
        ))
        fig_csyt.add_trace(go.Bar(
            x=top_csyt['Tên CSYT'],
            y=top_csyt['Tổng Giá trị cung cấp'],
            name='Đã cung cấp',
            marker_color='darkblue'
        ))
        
        fig_csyt.update_layout(
            title="Top 15 CSYT theo giá trị phân bổ",
            xaxis_tickangle=-45,
            barmode='group',
            height=500
        )
        st.plotly_chart(fig_csyt, use_container_width=True)

with tab2:
    if 'Nhóm' in df_filtered.columns:
        nhom_stats = df_filtered.groupby('Nhóm').agg({
            'Tổng giá trị được phân bổ': 'sum',
            'Tổng Giá trị cung cấp': 'sum'
        }).reset_index()
        
        nhom_stats['% Thực hiện'] = np.where(
            nhom_stats['Tổng giá trị được phân bổ'] > 0,
            (nhom_stats['Tổng Giá trị cung cấp'] / nhom_stats['Tổng giá trị được phân bổ'] * 100),
            0
        )
        
        fig_nhom = px.sunburst(
            nhom_stats,
            path=['Nhóm'],
            values='Tổng giá trị được phân bổ',
            color='% Thực hiện',
            color_continuous_scale='RdYlGn',
            title="Phân bố theo Nhóm thuốc"
        )
        st.plotly_chart(fig_nhom, use_container_width=True)

with tab3:
    # Cảnh báo tồn kho
    if 'Tên Hoạt chất' in df_filtered.columns:
        inventory_alert = df_filtered.groupby('Tên Hoạt chất').agg({
            'Tổng giá trị được phân bổ': 'sum',
            'Tổng Giá trị cung cấp': 'sum'
        }).reset_index()
        
        inventory_alert['% Thực hiện'] = np.where(
            inventory_alert['Tổng giá trị được phân bổ'] > 0,
            (inventory_alert['Tổng Giá trị cung cấp'] / inventory_alert['Tổng giá trị được phân bổ'] * 100),
            0
        )
        
        # Lọc các hoạt chất có giá trị lớn nhưng tỷ lệ thực hiện thấp
        low_performance = inventory_alert[
            (inventory_alert['Tổng giá trị được phân bổ'] > 100_000_000) &
            (inventory_alert['% Thực hiện'] < 20)
        ].sort_values('% Thực hiện')
        
        if not low_performance.empty:
            fig_alert = px.bar(
                low_performance.head(20),
                x='% Thực hiện',
                y='Tên Hoạt chất',
                orientation='h',
                title="⚠️ 20 Hoạt chất có tỷ lệ sử dụng thấp (< 20%) - Nguy cơ tồn kho",
                color='% Thực hiện',
                color_continuous_scale='Reds_r'
            )
            fig_alert.update_layout(height=600)
            st.plotly_chart(fig_alert, use_container_width=True)
        else:
            st.success("✅ Không có hoạt chất nào có nguy cơ tồn kho cao!")

st.markdown("---")

# ===========================
# BẢNG DỮ LIỆU CHI TIẾT
# ===========================
st.subheader("📋 Dữ liệu chi tiết")

if st.checkbox("📊 Hiển thị bảng dữ liệu", value=False):
    
    col_btn1, col_btn2 = st.columns([1, 4])
    
    with col_btn1:
        # Cho phép tải xuống
        csv = df_filtered.to_csv(index=False, encoding='utf-8-sig', sep=';')
        st.download_button(
            label="💾 Tải CSV",
            data=csv,
            file_name=f"du_lieu_loc_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_btn2:
        # Tùy chọn số dòng hiển thị
        rows_per_page = st.selectbox(
            "Số dòng mỗi trang:",
            options=[50, 100, 200, 500, 1000, len(df_filtered)],
            index=1,
            format_func=lambda x: f"{x:,} dòng" if x < len(df_filtered) else "Tất cả"
        )
    
    # Tính số cells
    total_cells = len(df_filtered) * len(df_filtered.columns)
    total_rows = len(df_filtered)
    
    st.info(f"📋 Tổng số: {total_rows:,} dòng × {len(df_filtered.columns)} cột")
    
    # Phân trang
    if rows_per_page < total_rows:
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        col_prev, col_page, col_next = st.columns([1, 3, 1])
        
        # Initialize session state for page number
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        with col_prev:
            if st.button("⬅️ Trang trước", disabled=(st.session_state.current_page == 1)):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col_page:
            page_num = st.selectbox(
                f"Trang (Tổng {total_pages} trang):",
                options=range(1, total_pages + 1),
                index=st.session_state.current_page - 1,
                key='page_selector'
            )
            if page_num != st.session_state.current_page:
                st.session_state.current_page = page_num
        
        with col_next:
            if st.button("Trang sau ➡️", disabled=(st.session_state.current_page == total_pages)):
                st.session_state.current_page += 1
                st.rerun()
        
        # Lấy dữ liệu trang hiện tại
        start_idx = (st.session_state.current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        df_page = df_filtered.iloc[start_idx:end_idx].copy()
        
        st.caption(f"Đang hiển thị dòng {start_idx + 1:,} - {end_idx:,}")
    else:
        df_page = df_filtered.copy()
    
    # Hiển thị bảng
    page_cells = len(df_page) * len(df_page.columns)
    
    if page_cells > 262144:
        st.warning("⚠️ Trang này có quá nhiều cells, hiển thị không có định dạng.")
        st.dataframe(df_page, use_container_width=True, height=500)
    else:
        # Format các cột số
        format_dict = {}
        if 'Tổng giá trị được phân bổ' in df_page.columns:
            format_dict['Tổng giá trị được phân bổ'] = '{:,.0f}'
        if 'Tổng Giá trị cung cấp' in df_page.columns:
            format_dict['Tổng Giá trị cung cấp'] = '{:,.0f}'
        if 'Giá trúng thầu' in df_page.columns:
            format_dict['Giá trúng thầu'] = '{:,.0f}'
        if 'Ty_le_thuc_hien' in df_page.columns:
            format_dict['Ty_le_thuc_hien'] = '{:.2f}%'
        
        st.dataframe(
            df_page.style.format(format_dict),
            use_container_width=True,
            height=500
        )

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>📊 Dashboard Phân tích Kết quả Thầu TTQG 2025 | Phiên bản 1.0</p>
    <p>Dữ liệu được cập nhật: {}</p>
</div>
""".format(pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')), unsafe_allow_html=True)