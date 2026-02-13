# ROS2 Humble with CUDA for vision-assisted navigation simulation
ARG BASE_IMAGE=althack/ros2:humble-cuda-full
FROM ${BASE_IMAGE}

# Build args: USERNAME is typically passed from devcontainer (${localEnv:USER})
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
    # GUI/display libs (X11, OpenGL) for simulation visualization
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       sudo git libgl1 libglib2.0-0 libsm6 libice6 libxrender1 libxext6 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/* \
    # Base image has pre-existing user (e.g. "ros") with UID 1000; reuse instead of creating new
    && EXISTING_USER=$(getent passwd ${USER_UID} | cut -d: -f1) \
    && if [ -n "$EXISTING_USER" ] && [ "$EXISTING_USER" != "${USERNAME}" ]; then \
         usermod -l ${USERNAME} $EXISTING_USER \
         && if [ -d "/home/$EXISTING_USER" ]; then usermod -d /home/${USERNAME} -m ${USERNAME}; fi; \
       elif [ -z "$EXISTING_USER" ]; then \
         (groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true) \
         && useradd -m -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}; \
       fi \
    # Passwordless sudo for dev workflows
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
