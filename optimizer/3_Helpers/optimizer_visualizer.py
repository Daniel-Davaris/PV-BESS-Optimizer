from IPython.display import display
import plotly.graph_objects as go
import pandas as pd
import time


def create_window_progress_visual(window_starts,total_dataset_intervals,intervals_per_bar,bars_per_window,retained_bars_per_window,dataset):
    date_series = pd.to_datetime(dataset["Date"]).reset_index(drop=True)
    fig = go.Figure()
    completed_x,completed_y,completed_colors,completed_real_dates,hover_pending = [],[],[],[],[]
    green_fill_x,green_fill_y = [],[]
    red_fill_x,red_fill_y = [],[]
    green_x,green_y,green_hover = [],[],[]
    dark_green_x,dark_green_y,dark_green_hover = [],[],[]
    red_x,red_y,red_hover = [],[],[]
    plot_year = 2000
    plot_month = 1
    operation_intervals_per_bar = max(int(intervals_per_bar),1)
    visual_bars_per_window = int(bars_per_window * operation_intervals_per_bar)
    visual_retained_bars_per_window = int(retained_bars_per_window * operation_intervals_per_bar)
    month_periods = sorted(date_series.dt.to_period("M").unique())
    month_labels = [period.strftime("%Y-%m") for period in month_periods]
    month_to_y = {period:i for i,period in enumerate(month_periods)}
    visual_row_gap = 0.13
    point_lookup = {}
    red_periods = set()

    for window,start_interval in enumerate(window_starts):
        start_bar = int(start_interval / intervals_per_bar)
        start_interval = int(start_bar * intervals_per_bar)
        if start_interval >= len(date_series): continue

        for bar in range(visual_bars_per_window):
            bar_start_interval = int(start_interval + bar)
            if bar_start_interval >= len(date_series): continue
            if bar >= visual_retained_bars_per_window: red_periods.add(date_series.iloc[bar_start_interval])

    interval_minutes = int(date_series.diff().dropna().dt.total_seconds().median() / 60) if len(date_series.diff().dropna()) else 60
    marker_size = max(3,min(9,int(360 / max((31 * 24 * 60 / interval_minutes),1))))
    legend_marker_size = max(marker_size + 10,16)
    visual_interval_delta = pd.Timedelta(minutes=interval_minutes)

    for window,start_interval in enumerate(window_starts):
        start_bar = int(start_interval / intervals_per_bar)
        start_interval = int(start_bar * intervals_per_bar)
        if start_interval >= len(date_series): continue
        current_fill_segment_start = None
        current_fill_segment_end = None
        current_fill_segment_y = None
        current_fill_segment_month = None
        current_fill_segment_is_overlap = None

        for bar in range(visual_bars_per_window):
            bar_start_interval = int(start_interval + bar)
            if bar_start_interval >= len(date_series): continue
            dt = date_series.iloc[bar_start_interval]
            month_period = dt.to_period("M")
            if month_period not in month_to_y: continue
            is_overlap = bar >= visual_retained_bars_per_window
            is_green_over_red = (not is_overlap) and dt in red_periods
            visual_dt = pd.Timestamp(year=plot_year,month=plot_month,day=dt.day,hour=dt.hour,minute=dt.minute,second=dt.second)
            window_y = month_to_y[month_period] + visual_row_gap if is_overlap else month_to_y[month_period] - visual_row_gap
            color = "rgba(255,80,80,0.98)" if is_overlap else "rgba(60,180,105,0.98)" if is_green_over_red else "rgba(120,255,160,0.98)"
            hover = f"Window: {window}<br>Interval: {bar}<br>Type: {'Discarded edge effect guard' if is_overlap else 'Retained output overlapping guard' if is_green_over_red else 'Retained output'}<br>Date: {dt}"
            point_lookup[(window,int(bar / operation_intervals_per_bar))] = len(completed_x)
            completed_x.append(visual_dt); completed_y.append(window_y); completed_real_dates.append(dt)
            completed_colors.append(color); hover_pending.append(hover)

            if current_fill_segment_start is None:
                current_fill_segment_start = visual_dt
                current_fill_segment_end = visual_dt + visual_interval_delta
                current_fill_segment_y = window_y
                current_fill_segment_month = month_period
                current_fill_segment_is_overlap = is_overlap
            elif current_fill_segment_y != window_y or current_fill_segment_month != month_period or current_fill_segment_is_overlap != is_overlap:
                if current_fill_segment_is_overlap:
                    red_fill_x.extend([current_fill_segment_start,current_fill_segment_end,current_fill_segment_end,current_fill_segment_start,current_fill_segment_start,None])
                    red_fill_y.extend([current_fill_segment_y - 0.31,current_fill_segment_y - 0.31,current_fill_segment_y + 0.31,current_fill_segment_y + 0.31,current_fill_segment_y - 0.31,None])
                else:
                    green_fill_x.extend([current_fill_segment_start,current_fill_segment_end,current_fill_segment_end,current_fill_segment_start,current_fill_segment_start,None])
                    green_fill_y.extend([current_fill_segment_y - 0.31,current_fill_segment_y - 0.31,current_fill_segment_y + 0.31,current_fill_segment_y + 0.31,current_fill_segment_y - 0.31,None])

                current_fill_segment_start = visual_dt
                current_fill_segment_end = visual_dt + visual_interval_delta
                current_fill_segment_y = window_y
                current_fill_segment_month = month_period
                current_fill_segment_is_overlap = is_overlap
            else:
                current_fill_segment_end = visual_dt + visual_interval_delta

            if is_overlap:
                red_x.append(visual_dt); red_y.append(window_y); red_hover.append(hover)
            elif is_green_over_red:
                dark_green_x.append(visual_dt); dark_green_y.append(window_y); dark_green_hover.append(hover)
            else:
                green_x.append(visual_dt); green_y.append(window_y); green_hover.append(hover)

        if current_fill_segment_start is not None:
            if current_fill_segment_is_overlap:
                red_fill_x.extend([current_fill_segment_start,current_fill_segment_end,current_fill_segment_end,current_fill_segment_start,current_fill_segment_start,None])
                red_fill_y.extend([current_fill_segment_y - 0.31,current_fill_segment_y - 0.31,current_fill_segment_y + 0.31,current_fill_segment_y + 0.31,current_fill_segment_y - 0.31,None])
            else:
                green_fill_x.extend([current_fill_segment_start,current_fill_segment_end,current_fill_segment_end,current_fill_segment_start,current_fill_segment_start,None])
                green_fill_y.extend([current_fill_segment_y - 0.31,current_fill_segment_y - 0.31,current_fill_segment_y + 0.31,current_fill_segment_y + 0.31,current_fill_segment_y - 0.31,None])

    day_starts = pd.date_range(start=pd.Timestamp(f"{plot_year}-{plot_month:02d}-01"),end=pd.Timestamp(f"{plot_year}-{plot_month:02d}-31"),freq="1D")
    background_shapes = [dict(type="line",xref="x",yref="paper",x0=dt,x1=dt,y0=0,y1=1,line=dict(color="rgba(255,255,255,0.10)",width=1),layer="below") for dt in day_starts]

    fig.add_trace(go.Scatter(x=green_fill_x,y=green_fill_y,mode="lines",fill="toself",fillcolor="rgba(0,190,95,0.22)",line=dict(color="rgba(0,210,105,0.75)",width=1.2),hoverinfo="skip",name="Retained window",showlegend=True))
    fig.add_trace(go.Scatter(x=red_fill_x,y=red_fill_y,mode="lines",fill="toself",fillcolor="rgba(255,60,60,0.22)",line=dict(color="rgba(255,80,80,0.75)",width=1.2),hoverinfo="skip",name="Discarded window",showlegend=True))
    fig.add_trace(go.Scattergl(x=green_x,y=green_y,mode="markers",marker=dict(size=marker_size,color="rgba(120,255,160,0.98)",line=dict(color="rgba(0,0,0,1)",width=0.5),symbol="square"),hovertext=green_hover,hoverinfo="text",name="Retained output",showlegend=True))
    fig.add_trace(go.Scattergl(x=dark_green_x,y=dark_green_y,mode="markers",marker=dict(size=marker_size,color="rgba(60,180,105,0.98)",line=dict(color="rgba(0,0,0,1)",width=0.5),symbol="square"),hovertext=dark_green_hover,hoverinfo="text",name="Retained output overlapping guard",showlegend=True))
    fig.add_trace(go.Scattergl(x=red_x,y=red_y,mode="markers",marker=dict(size=marker_size,color="rgba(255,80,80,0.98)",line=dict(color="rgba(0,0,0,1)",width=0.5),symbol="square"),hovertext=red_hover,hoverinfo="text",name="Discarded edge effect guard",showlegend=True))
    fig.add_trace(go.Scattergl(x=[],y=[],mode="markers",marker=dict(size=legend_marker_size,color="rgba(255,80,180,1)",line=dict(color="rgba(0,0,0,1)",width=1.8),symbol="square"),hoverinfo="skip",name="Current horizon",showlegend=True))

    fig.update_layout(
        template="plotly_dark",autosize=True,height=max(420,62 * len(month_periods)),plot_bgcolor="#111111",paper_bgcolor="#111111",showlegend=True,shapes=background_shapes,
        margin=dict(l=75,r=10,t=90,b=35),
        title=dict(text="Rolling Optimization Window Evolution",x=0.5,y=0.97,xanchor="center",font=dict(size=21,color="rgba(235,235,235,1)")),
        legend=dict(orientation="h",yanchor="top",y=1.05,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)",font=dict(size=11),itemsizing="constant"),
        xaxis=dict(title="Day of month",range=[pd.Timestamp(f"{plot_year}-{plot_month:02d}-01"),pd.Timestamp(f"{plot_year}-{plot_month:02d}-31 23:59")],showgrid=False,zeroline=False,tickmode="array",tickvals=list(day_starts),ticktext=[str(dt.day) for dt in day_starts],tickfont=dict(size=10),showline=True,linewidth=1,linecolor="rgba(255,255,255,0.35)",domain=[0,1],rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=True,visible=True,tickmode="array",tickvals=list(month_to_y.values()),ticktext=month_labels,tickfont=dict(size=11),range=[-0.45,len(month_periods) - 1 + 0.45],domain=[0,1],fixedrange=True)
    )

    fig._last_visual_update = 0
    fig._completed_real_dates = completed_real_dates
    fig._simulation_start_date = date_series.iloc[0]
    fig._simulation_end_date = date_series.iloc[-1]
    fig._total_simulation_days = (date_series.iloc[-1].normalize() - date_series.iloc[0].normalize()).days + 1
    fig._point_lookup = point_lookup
    display(fig,config={"displayModeBar":False,"responsive":True})
    return fig,completed_x,completed_y,completed_colors


def update_window_progress_visual(fig,current_window,current_horizon_bar,window_starts,bars_per_window,completed_x,completed_y,completed_colors,force=False,min_update_seconds=0.25):
    now = time.time()
    if not force and now - getattr(fig,"_last_visual_update",0) < min_update_seconds: return
    fig._last_visual_update = now
    point_lookup = getattr(fig,"_point_lookup",{})
    total_available_points = len(completed_x)
    current_point_index = point_lookup.get((current_window,current_horizon_bar),min((current_window * bars_per_window) + current_horizon_bar,total_available_points - 1))
    real_dates = getattr(fig,"_completed_real_dates",[])
    total_simulation_days = getattr(fig,"_total_simulation_days",total_available_points)
    simulation_start_date = getattr(fig,"_simulation_start_date",real_dates[0] if real_dates else None)
    simulation_day = min((real_dates[current_point_index].normalize() - simulation_start_date.normalize()).days + 1,total_simulation_days) if real_dates and simulation_start_date is not None else current_point_index + 1

    with fig.batch_update():
        fig.data[5].x = [completed_x[current_point_index]]
        fig.data[5].y = [completed_y[current_point_index]]
        fig.layout.title.text = f"Optimization | Window {current_window + 1}/{len(window_starts)} | Day {simulation_day}/{total_simulation_days}"