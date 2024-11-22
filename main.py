import pygame
from video_processing import play_video, play_webcam
from utils import get_video_file

# Global variables for screen dimensions
screen_width = 800
screen_height = 600

# Colors
COLOR_BACKGROUND = (20, 20, 20)
COLOR_BUTTON = (50, 50, 50)
COLOR_BUTTON_HOVER = (70, 70, 70)
COLOR_TEXT = (255, 255, 255)
COLOR_TITLE = (200, 200, 200)
COLOR_EXIT_BUTTON = (200, 0, 0)
COLOR_EXIT_BUTTON_HOVER = (255, 0, 0)
COLOR_CHECKBOX = (100, 100, 100)
COLOR_CHECKBOX_CHECKED = (0, 200, 0)

def draw_button(screen, rect, text, font, is_hovered=False):
    color = COLOR_BUTTON_HOVER if is_hovered else COLOR_BUTTON
    pygame.draw.rect(screen, color, rect)
    text_surf = font.render(text, True, COLOR_TEXT)
    screen.blit(text_surf, (rect.x + (rect.width - text_surf.get_width()) // 2, rect.y + (rect.height - text_surf.get_height()) // 2))

def draw_checkbox(screen, rect, is_checked, font, label):
    pygame.draw.rect(screen, COLOR_CHECKBOX, rect)
    if is_checked:
        pygame.draw.line(screen, COLOR_CHECKBOX_CHECKED, (rect.x + 5, rect.y + 5), (rect.x + rect.width - 5, rect.y + rect.height - 5), 2)
        pygame.draw.line(screen, COLOR_CHECKBOX_CHECKED, (rect.x + rect.width - 5, rect.y + 5), (rect.x + 5, rect.y + rect.height - 5), 2)
    text_surf = font.render(label, True, COLOR_TEXT)
    screen.blit(text_surf, (rect.x + rect.width + 10, rect.y + (rect.height - text_surf.get_height()) // 2))

def main_menu():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Select Input')

    font = pygame.font.Font(None, 36)
    title_font = pygame.font.Font(None, 48)
    title = "Lie Detector v5"

    # Button definitions
    button_width = 200
    button_height = 50
    button_spacing = 20
    start_y = 150
    title_y = 50

    webcam_button = pygame.Rect((screen_width - button_width) // 2, start_y, button_width, button_height)
    video_button = pygame.Rect((screen_width - button_width) // 2, start_y + button_height + button_spacing, button_width, button_height)
    settings_checkbox = pygame.Rect((screen_width - button_width) // 2, start_y + 2 * (button_height + button_spacing), 30, 30)
    exit_button = pygame.Rect((screen_width - button_width) // 2, start_y + 3 * (button_height + button_spacing), button_width, button_height)

    draw_landmarks = False  # Default setting for landmark drawing

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos() #cursor position
        mouse_click = pygame.mouse.get_pressed() #stateof the mouse buttons

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if webcam_button.collidepoint(event.pos):
                    play_webcam(screen, draw_landmarks)
                    screen = pygame.display.set_mode((screen_width, screen_height))  # Reinitialize Pygame display after exiting playback
                if video_button.collidepoint(event.pos):
                    video_file = get_video_file()
                    if video_file:
                        play_video(video_file, screen, draw_landmarks)
                        screen = pygame.display.set_mode((screen_width, screen_height))  # Reinitialize Pygame display after exiting playback
                if settings_checkbox.collidepoint(event.pos):
                    draw_landmarks = not draw_landmarks  # Toggle landmark drawing
                if exit_button.collidepoint(event.pos):
                    running = False

        screen.fill(COLOR_BACKGROUND)
        title_text = title_font.render(title, True, COLOR_TITLE)
        screen.blit(title_text, (screen_width // 2 - title_text.get_width() // 2, title_y))

        draw_button(screen, webcam_button, 'Webcam', font, webcam_button.collidepoint(mouse_pos))
        draw_button(screen, video_button, 'Video File', font, video_button.collidepoint(mouse_pos))
        draw_checkbox(screen, settings_checkbox, draw_landmarks, font, 'Draw Landmarks')
        draw_button(screen, exit_button, 'Exit', font, exit_button.collidepoint(mouse_pos))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main_menu()
