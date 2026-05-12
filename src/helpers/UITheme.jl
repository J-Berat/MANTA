# Shared Makie UI theme for MANTA viewers.

struct MANTAUITheme
    accent::RGBf
    accent_dim::RGBf
    accent_strong::RGBf
    track::RGBf
    surface::RGBf
    surface_hover::RGBf
    surface_active::RGBf
    panel::RGBf
    panel_header::RGBf
    border::RGBf
    border_strong::RGBf
    text::RGBf
    text_muted::RGBf
    background::RGBf
end

default_ui_theme() = MANTAUITheme(
    RGBf(0.36, 0.39, 0.92),    # indigo-500
    RGBf(0.62, 0.64, 0.96),    # indigo-300
    RGBf(0.28, 0.31, 0.82),    # indigo-700
    RGBf(0.88, 0.90, 0.95),    # slate-200
    RGBf(0.985, 0.988, 0.996), # near-white card
    RGBf(0.94, 0.95, 0.99),
    RGBf(0.90, 0.92, 0.98),
    RGBf(0.965, 0.970, 0.985),
    RGBf(0.93, 0.94, 0.97),
    RGBf(0.78, 0.81, 0.88),
    RGBf(0.62, 0.66, 0.76),
    RGBf(0.10, 0.12, 0.20),
    RGBf(0.42, 0.46, 0.56),
    RGBf(0.97, 0.975, 0.985),
)

function manta_style_checkbox!(chk, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    chk.size[] = compact ? 18 : 22
    chk.checkmarksize[] = compact ? 0.58 : 0.62
    chk.roundness[] = 0.5
    chk.checkboxstrokewidth[] = 1.4
    chk.checkboxcolor_checked[] = theme.accent
    chk.checkboxcolor_unchecked[] = RGBf(0.96, 0.965, 0.985)
    chk.checkboxstrokecolor_checked[] = theme.accent_strong
    chk.checkboxstrokecolor_unchecked[] = theme.border
    chk.checkmarkcolor_checked[] = :white
    chk.checkmarkcolor_unchecked[] = RGBf(0.65, 0.70, 0.78)
    chk
end

function manta_style_slider!(sl, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    sl.height[] = compact ? 20 : 26
    sl.linewidth[] = compact ? 8 : 10
    sl.color_active[] = theme.accent
    sl.color_active_dimmed[] = theme.accent_dim
    sl.color_inactive[] = theme.track
    sl
end

function manta_style_button!(btn, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    btn.height[] = compact ? 30 : 34
    btn.cornerradius[] = 8
    btn.strokewidth[] = 1.0
    btn.strokecolor[] = theme.border
    btn.buttoncolor[] = theme.surface
    btn.buttoncolor_hover[] = theme.surface_hover
    btn.buttoncolor_active[] = theme.surface_active
    btn.labelcolor[] = theme.text
    btn.labelcolor_hover[] = theme.accent_strong
    btn.labelcolor_active[] = theme.accent_strong
    btn.fontsize[] = compact ? 13 : 14
    btn.padding[] = compact ? (9, 9, 5, 5) : (12, 12, 7, 7)
    btn
end

function manta_style_menu!(menu, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    menu.height[] = compact ? 30 : 34
    menu.width[] = max(menu.width[], 96)
    menu.textcolor[] = theme.text
    menu.fontsize[] = compact ? 13 : 14
    menu.dropdown_arrow_color[] = theme.accent
    menu.dropdown_arrow_size[] = compact ? 10 : 11
    menu.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    menu.cell_color_inactive_even[] = theme.surface
    menu.cell_color_inactive_odd[] = theme.surface
    menu.selection_cell_color_inactive[] = theme.surface
    menu.cell_color_hover[] = theme.surface_hover
    menu.cell_color_active[] = theme.surface_active
    menu
end

function manta_style_textbox!(tb, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    tb.height[] = compact ? 30 : 34
    tb.fontsize[] = compact ? 13 : 14
    tb.textcolor[] = theme.text
    tb.textcolor_placeholder[] = theme.text_muted
    tb.boxcolor[] = theme.surface
    tb.boxcolor_hover[] = theme.surface_hover
    tb.boxcolor_focused[] = RGBf(1.0, 1.0, 1.0)
    tb.bordercolor[] = theme.border
    tb.bordercolor_hover[] = theme.accent_dim
    tb.bordercolor_focused[] = theme.accent
    tb.borderwidth[] = 1.4
    tb.cornerradius[] = 8
    tb.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    tb
end
