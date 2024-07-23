import mpv

try:
    player = mpv.MPV(ytdl=True)
    print("mpv initialized successfully")
except Exception as e:
    print(f"Failed to initialize mpv: {e}")