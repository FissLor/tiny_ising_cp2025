/*
 Copyright (c) 2021 Carlos Bederián <carlos.bederian@unc.edu.ar>
 License: http://www.gnu.org/licenses/gpl-2.0.html
*/

#pragma once

#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Interface */
typedef struct _gl2d_s *gl2d_t;

static gl2d_t gl2d_init(const char *title, int width, int height);
static void gl2d_destroy(gl2d_t gl2d);
static void gl2d_draw_rgb888(gl2d_t gl2d, int offset_x, int offset_y, int width, int height, const uint8_t *rgb888_data);
static void gl2d_draw_rgbf(gl2d_t gl2d, int offset_x, int offset_y, int width, int height, const float *rgbf_data);
static void gl2d_display(gl2d_t gl2d);
static int gl2d_should_close(gl2d_t gl2d);


/* Implementation */
struct _gl2d_s {
    int width, height;
    GLFWwindow *window;
    GLuint tex_name;
};


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static void gl2d_draw_rgb888(gl2d_t gl2d, int offset_x, int offset_y, int width, int height, const uint8_t *rgb888_data) {
    glBindTexture(GL_TEXTURE_2D, gl2d->tex_name);
    glTexSubImage2D(GL_TEXTURE_2D, 0, offset_x, offset_y, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb888_data);
    glBindTexture(GL_TEXTURE_2D, 0);
}


static void gl2d_draw_rgbf(gl2d_t gl2d, int offset_x, int offset_y, int width, int height, const float *rgbf_data) {
    glBindTexture(GL_TEXTURE_2D, gl2d->tex_name);
    glTexSubImage2D(GL_TEXTURE_2D, 0, offset_x, offset_y, width, height, GL_RGB, GL_FLOAT, rgbf_data);
    glBindTexture(GL_TEXTURE_2D, 0);
}


static void gl2d_display(gl2d_t gl2d) {
    // clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // select our texture
    glBindTexture(GL_TEXTURE_2D, gl2d->tex_name);

    // place a textured quad
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(0.0f, 1.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(1.0f, 1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    // bring backbuffer to front
    glfwSwapBuffers(gl2d->window);

    // Process resize events
    glfwPollEvents();
}


static void gl2d__error_callback(int error, const char* description) {
    fprintf(stderr, "GL Error %d: %s\n", error, description);
}


static void gl2d__change_window_size(int width, int height) {
    // Set up OpenGL projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
}


static void gl2d__framebuffer_size_callback(__attribute__((unused)) GLFWwindow *window, int width, int height) {
    gl2d__change_window_size(width, height);
}


static gl2d_t gl2d_init(const char *title, int width, int height) {

    if (title == NULL) {
        // TODO: Error
        return NULL;
    }

    if ((width <= 0) || (width > GL_MAX_TEXTURE_SIZE) || (height <= 0) || (height > GL_MAX_TEXTURE_SIZE)) {
        // TODO: Error
        return NULL;
    }

    glfwSetErrorCallback(gl2d__error_callback);

    if (!glfwInit()) {
        // TODO: Error
        return NULL;
    }

    gl2d_t gl2d = calloc(1, sizeof(struct _gl2d_s));
    gl2d->window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (gl2d->window == NULL) {
        // TODO: Error
        glfwTerminate();
        free(gl2d);
        return NULL;
    }

    gl2d->width = width;
    gl2d->height = height;

    glfwMakeContextCurrent(gl2d->window);
    // TODO: Load GL extensions?

    // vsync
    glfwSwapInterval(1);

    // enable texturing
    glEnable(GL_TEXTURE_2D);
    // disable depth
    glDisable(GL_DEPTH_TEST);

    // create and set up our texture
    glGenTextures(1, &gl2d->tex_name);
    glBindTexture(GL_TEXTURE_2D, gl2d->tex_name);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Set up window size
    gl2d__change_window_size(gl2d->width, gl2d->height);
    glfwSetFramebufferSizeCallback(gl2d->window, gl2d__framebuffer_size_callback);

    // Clear
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    return gl2d;
}


static int gl2d_should_close(gl2d_t gl2d) {
    return glfwWindowShouldClose(gl2d->window);
}


static void gl2d_destroy(gl2d_t gl2d) {
    glDeleteTextures(1, &gl2d->tex_name);
    glfwDestroyWindow(gl2d->window);
    glfwTerminate();
    free(gl2d);
}
#pragma GCC diagnostic pop
