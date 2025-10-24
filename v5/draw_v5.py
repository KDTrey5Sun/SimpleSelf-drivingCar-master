import os
import pygame

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imgs')
TRACK_PATH = os.path.join(IMG_DIR, 'track1.png')
BORDER_PATH = os.path.join(IMG_DIR, 'track_border1.png')


def scale_surface(surf: pygame.Surface, scale: float) -> pygame.Surface:
	if scale == 1.0:
		return surf.copy()
	w, h = surf.get_size()
	new_size = (int(w * scale), int(h * scale))
	return pygame.transform.smoothscale(surf, new_size)


def main():
	pygame.init()
	# 加载图片（原始尺寸）
	track_orig = pygame.image.load(TRACK_PATH)
	border_orig = pygame.image.load(BORDER_PATH)

	# 默认使用与训练一致的缩放 0.3，可按键切换
	scale = 0.3

	def build_window_and_surfaces(s: float):
		track_s = scale_surface(track_orig, s)
		border_s = scale_surface(border_orig, s)
		w, h = track_s.get_size()
		win_local = pygame.display.set_mode((w, h))
		pygame.display.set_caption(
			f'Track Viewer (B:toggle border, +/-:scale, 1=1.0, 3=0.3, S:save)  scale={s:.2f}'
		)
		return win_local, track_s, border_s

	win, track, border = build_window_and_surfaces(scale)

	clock = pygame.time.Clock()
	show_border = True
	running = True
	while running:
		clock.tick(60)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_b:
					show_border = not show_border
				elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
					scale = min(2.0, scale + 0.05)
					win, track, border = build_window_and_surfaces(scale)
				elif event.key == pygame.K_MINUS:
					scale = max(0.05, scale - 0.05)
					win, track, border = build_window_and_surfaces(scale)
				elif event.key == pygame.K_1:
					scale = 1.0
					win, track, border = build_window_and_surfaces(scale)
				elif event.key == pygame.K_3:
					scale = 0.3
					win, track, border = build_window_and_surfaces(scale)
				elif event.key == pygame.K_s:
					out = os.path.join(IMG_DIR, 'track_view_export.png')
					pygame.image.save(win, out)
					print('Saved:', out)

		win.blit(track, (0, 0))
		if show_border:
			win.blit(border, (0, 0))
		pygame.display.update()

	pygame.quit()


if __name__ == '__main__':
	main()

