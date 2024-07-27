#include <WiFi.h>
#include <WiFiClient.h>
#include <ESPAsyncWebServer.h>
#include "esp_camera.h"
#include "Freenove_4WD_Car_For_ESP32.h"

#define CAMERA_MODEL_WROVER_KIT
#include "camera_pins.h"

const char* ssid = "PHYSICS4AL/4BL";
const char* password = "physics4";
WiFiServer tcpServer(8080);
AsyncWebServer httpServer(80);

void startCameraServer();

void setup() {
    // Initialize Serial for debugging
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    // Initialize WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to the WiFi network");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Initialize PCA9685 for motor control
    PCA9685_Setup();

    // Camera configuration
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size = FRAMESIZE_UXGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
    } else {
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return; // Stop further execution if camera init fails
    }
    sensor_t * s = esp_camera_sensor_get();
    s->set_framesize(s, FRAMESIZE_VGA);

    startCameraServer();
    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.println("' to connect");

    httpServer.begin();
    Serial.println("HTTP server started");
    tcpServer.begin();
}

void loop() {
    WiFiClient client = tcpServer.available();
    if (client) {
        Serial.println("Client connected");
        while (client.connected()) {
            if (client.available()) {
                String command = client.readStringUntil('\n');
                command.trim(); // Remove any whitespace
                Serial.println("Received command: " + command);
                processCommand(command);
                // Send acknowledgment
                client.println("ACK");
            }
        }
        client.stop();
        Serial.println("Client disconnected");
    }
}

void processCommand(const String& command) {
    // Split the command string by spaces
    int spaceIndex = command.indexOf(' ');
    if (spaceIndex == -1) {
        Serial.println("Invalid command format");
        return;
    }

    // Extract the command
    String cmd = command.substring(0, spaceIndex);
    Serial.print("Command: "); Serial.println(cmd);

    // Extract the speed values
    int speeds[4] = {0, 0, 0, 0};
    int prevIndex = spaceIndex + 1;
    for (int i = 0; i < 4; i++) {
        spaceIndex = command.indexOf(' ', prevIndex);
        if (spaceIndex == -1 && i < 3) {
            Serial.println("Invalid command format");
            return;
        }
        speeds[i] = command.substring(prevIndex, spaceIndex).toInt();
        prevIndex = spaceIndex + 1;
    }

    Serial.print("Speeds: ");
    for (int i = 0; i < 4; i++) {
        Serial.print(speeds[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Execute the command using Motor_Move
    if (cmd == "FORWARD") {
        Serial.println("Moving forward");
        Motor_Move(speeds[0], speeds[1], speeds[2], speeds[3]);
    } else if (cmd == "BACKWARD") {
        Serial.println("Moving backward");
        Motor_Move(speeds[0], speeds[1], speeds[2], speeds[3]);
    } else if (cmd == "TURN_LEFT") {
        Serial.println("Turning left");
        Motor_Move(speeds[0], speeds[1], speeds[2], speeds[3]);
    } else if (cmd == "TURN_RIGHT") {
        Serial.println("Turning right");
        Motor_Move(speeds[0], speeds[1], speeds[2], speeds[3]);
    } else if (cmd == "STOP") {
        Serial.println("Stopping");
        Motor_Move(0, 0, 0, 0);
    } else if (cmd == "TURN") {
        Serial.println("Turning");
        Motor_Move(1000, 1000, -1000, -1000); // Example turn command
    } else {
        Serial.println("Unknown command");
    }
}

