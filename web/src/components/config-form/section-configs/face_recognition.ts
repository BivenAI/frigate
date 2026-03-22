import type { SectionConfigOverrides } from "./types";

const faceRecognition: SectionConfigOverrides = {
  base: {
    sectionDocs: "/configuration/face_recognition",
    restartRequired: [],
    fieldOrder: ["enabled", "min_area"],
    hiddenFields: [],
    advancedFields: [],
    overrideFields: ["enabled", "min_area"],
  },
  global: {
    fieldOrder: [
      "enabled",
      "detector",
      "model_size",
      "unknown_score",
      "detection_threshold",
      "recognition_threshold",
      "min_area",
      "min_faces",
      "save_attempts",
      "blur_confidence_filter",
      "device",
    ],
    advancedFields: [
      "detector",
      "unknown_score",
      "detection_threshold",
      "recognition_threshold",
      "min_area",
      "min_faces",
      "save_attempts",
      "blur_confidence_filter",
      "device",
    ],
    restartRequired: ["enabled", "detector", "model_size", "device"],
  },
};

export default faceRecognition;
