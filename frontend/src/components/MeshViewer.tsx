/**
 * Vanilla three.js mesh viewer. No drei / no react-three-fiber so it works
 * cleanly under React 19. Loads a .ply and frames the camera to it (the mesh
 * is kept in real coordinates so an editable placement transform — rotate /
 * move / scale, applied about the mesh centre — previews exactly what gets
 * baked into the file). Gives the user OrbitControls + a grid floor + lights.
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import { cn } from "@/lib/utils";

export type Placement = {
  rx: number; // rotation degrees about X
  ry: number;
  rz: number;
  tx: number; // translation (world units)
  ty: number;
  tz: number;
  scale: number; // uniform scale (>0)
};

export const IDENTITY_PLACEMENT: Placement = {
  rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0, scale: 1,
};

type LoadState = { state: "loading" | "ready" | "error"; msg?: string };

/** Placement matrix that rotates & scales about `center`, then translates. */
function placementMatrix(p: Placement, center: THREE.Vector3): THREE.Matrix4 {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(p.rx),
    THREE.MathUtils.degToRad(p.ry),
    THREE.MathUtils.degToRad(p.rz),
    "XYZ",
  );
  const R = new THREE.Matrix4().makeRotationFromEuler(euler);
  const S = new THREE.Matrix4().makeScale(p.scale, p.scale, p.scale);
  const toOrigin = new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z);
  const back = new THREE.Matrix4().makeTranslation(center.x, center.y, center.z);
  const T = new THREE.Matrix4().makeTranslation(p.tx, p.ty, p.tz);
  // M = T · back · R · S · toOrigin  (apply rightmost first)
  return new THREE.Matrix4()
    .multiply(T)
    .multiply(back)
    .multiply(R)
    .multiply(S)
    .multiply(toOrigin);
}

/** One agent to animate inside the mesh: sim-frame trajectory + the build's
 *  raw-mesh→sim 4×4 transform (so we can map points back onto the mesh). */
export type RobotReplay = {
  key: string;
  color: number; // three.js hex
  points: { x: number; y: number }[];
  goal?: { x: number; y: number }; // true goal (paths stop at the success radius)
  rawToSim: number[][];
  floorZ: number;
};

export function MeshViewer({
  url,
  className,
  placement = IDENTITY_PLACEMENT,
  onMatrix,
  robots = [],
}: {
  url: string;
  className?: string;
  placement?: Placement;
  onMatrix?: (elements: number[]) => void;
  robots?: RobotReplay[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [load, setLoad] = useState<LoadState>({ state: "loading" });
  // Live handles shared between the load effect and the placement effect.
  const envRef = useRef<{ group: THREE.Group; center: THREE.Vector3 } | null>(null);
  const placementRef = useRef(placement);
  placementRef.current = placement;
  const applyPlacementRef = useRef<(() => void) | null>(null);
  const onMatrixRef = useRef(onMatrix);
  onMatrixRef.current = onMatrix;
  const robotsPropRef = useRef(robots);
  robotsPropRef.current = robots;
  const applyRobotRef = useRef<(() => void) | null>(null);
  const robotsRef = useRef<
    Array<{
      points: THREE.Vector3[];
      robot: THREE.Mesh;
      frame: number;
      maxLen: number;
      speed: number;
      pauseEnd: number;
      radius: number;
      up: THREE.Vector3;
    }>
  >([]);

  // Scene setup + mesh load — re-runs only when the URL changes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !url) return;
    setLoad({ state: "loading" });
    envRef.current = null;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 2000);
    camera.position.set(4, 3, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.95);
    sun.position.set(8, 12, 6);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.25);
    fill.position.set(-6, 4, -5);
    scene.add(fill);

    const grid = new THREE.GridHelper(40, 40, 0x2a2a2a, 0x161616);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.7;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const envGroup = new THREE.Group();
    envGroup.matrixAutoUpdate = false;
    scene.add(envGroup);

    // Robot + path live inside envGroup so the placement transform applies to
    // them exactly like the mesh — they stay glued to the scanned room.
    const robotGroup = new THREE.Group();
    envGroup.add(robotGroup);

    // Held so applyRobots() can fade the room to a translucent "dollhouse" while
    // robots are navigating (so you can see them inside a closed scan), and
    // restore it to solid for the placement-editing stage.
    let meshMat: THREE.MeshStandardMaterial | null = null;

    const loader = new PLYLoader();
    loader.load(
      url,
      (geom) => {
        if (!geom.getAttribute("position")?.count) {
          setLoad({ state: "error", msg: "The 3D scene came out empty. Try a clip with more overlap and detail." });
          return;
        }
        geom.computeVertexNormals();
        geom.computeBoundingBox();
        geom.computeBoundingSphere();
        const center = geom.boundingBox!.getCenter(new THREE.Vector3());
        const radius = geom.boundingSphere?.radius || 1;

        const material = new THREE.MeshStandardMaterial({
          color: 0xdcdcdc,
          roughness: 0.55,
          metalness: 0.05,
          side: THREE.DoubleSide,
          vertexColors: !!geom.getAttribute("color"),
        });
        meshMat = material;
        const mesh = new THREE.Mesh(geom, material);
        envGroup.add(mesh);
        envRef.current = { group: envGroup, center };

        // Frame the camera to the mesh and orbit around its centre.
        const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2)) * 1.3;
        camera.position.copy(center).add(new THREE.Vector3(1, 0.7, 1).normalize().multiplyScalar(dist));
        camera.far = Math.max(2000, dist * 50);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();

        applyPlacement();
        applyRobots();
        setLoad({ state: "ready" });
      },
      undefined,
      (err) => {
        console.error("PLY load failed", err);
        setLoad({ state: "error", msg: "Couldn't load the 3D scene — it may be missing or failed to generate." });
      },
    );

    // Applies the latest placement prop to the loaded group + reports the matrix.
    function applyPlacement() {
      const env = envRef.current;
      if (!env) return;
      const M = placementMatrix(placementRef.current, env.center);
      env.group.matrix.copy(M);
      env.group.matrixWorldNeedsUpdate = true;
      onMatrixRef.current?.(M.elements.slice());
    }
    applyPlacementRef.current = applyPlacement;

    // Build one rolling ball + path per agent, mapping each sim-frame
    // trajectory back into raw-mesh coordinates via the inverse raw→sim
    // transform. All agents share the same start→goal (the compared config).
    function applyRobots() {
      while (robotGroup.children.length) {
        const c = robotGroup.children.pop() as THREE.Mesh | THREE.Line;
        (c as THREE.Mesh).geometry?.dispose?.();
        const mat = (c as THREE.Mesh).material as THREE.Material | undefined;
        mat?.dispose?.();
      }
      robotsRef.current = [];
      // Fade the room while robots navigate so they're visible inside a closed
      // scan; restore it solid when there are none (placement-editing stage).
      const hasRobots = (robotsPropRef.current ?? []).some((r) => r?.points?.length);
      if (meshMat) {
        meshMat.transparent = hasRobots;
        meshMat.opacity = hasRobots ? 0.45 : 1;
        meshMat.depthWrite = !hasRobots;
        meshMat.needsUpdate = true;
      }
      const agents = (robotsPropRef.current ?? []).filter(
        (r) => r?.points?.length && r.rawToSim?.length === 4,
      );
      let goalDone = false;
      let ai = 0;
      for (const rp of agents) {
        const m = rp.rawToSim;
        const M = new THREE.Matrix4().set(
          m[0][0], m[0][1], m[0][2], m[0][3],
          m[1][0], m[1][1], m[1][2], m[1][3],
          m[2][0], m[2][1], m[2][2], m[2][3],
          m[3][0], m[3][1], m[3][2], m[3][3],
        );
        const simToRaw = M.clone().invert();
        const sc = new THREE.Vector3().setFromMatrixScale(simToRaw);
        const up = new THREE.Vector3(0, 0, 1).transformDirection(simToRaw).normalize();
        // Physical agent radius is ~0.08m; render a touch larger (roomba-scale)
        // so both balls read clearly inside the room. Rest them on the floor.
        const VIS_R = 0.13;
        const robotR = Math.max(0.02, VIS_R * sc.x);
        const zRobot = rp.floorZ + VIS_R + 0.01;
        // Both policies share the same start→goal, so their paths often nearly
        // coincide. Slide each into its own lane (offset perpendicular to the
        // start→goal line, in the floor plane) so both balls stay distinct.
        const first = rp.points[0];
        const last = rp.points[rp.points.length - 1];
        let nx = -(last.y - first.y);
        let ny = last.x - first.x;
        const nlen = Math.hypot(nx, ny) || 1;
        nx /= nlen;
        ny /= nlen;
        const lane = (ai - (agents.length - 1) / 2) * (VIS_R * 1.7);
        const pts = rp.points.map((p) =>
          new THREE.Vector3(p.x + nx * lane, p.y + ny * lane, zRobot).applyMatrix4(simToRaw),
        );
        ai += 1;
        robotGroup.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: rp.color, transparent: true, opacity: 0.9 }),
          ),
        );
        if (!goalDone) {
          const goal = new THREE.Mesh(
            new THREE.SphereGeometry(robotR * 1.0, 16, 12),
            new THREE.MeshStandardMaterial({ color: 0xff5a3c, emissive: 0xff3a1e, emissiveIntensity: 0.6 }),
          );
          // Shared goal: the TRUE goal position (paths stop at the success
          // radius, short of it), on the centerline (no lane offset).
          const g = rp.goal ?? last;
          goal.position.copy(new THREE.Vector3(g.x, g.y, zRobot).applyMatrix4(simToRaw));
          robotGroup.add(goal);
          goalDone = true;
        }
        const robotMesh = new THREE.Mesh(
          new THREE.SphereGeometry(robotR, 24, 18),
          new THREE.MeshStandardMaterial({ color: rp.color, emissive: rp.color, emissiveIntensity: 0.6 }),
        );
        const cap = new THREE.Mesh(
          new THREE.SphereGeometry(robotR * 0.42, 10, 8),
          new THREE.MeshStandardMaterial({ color: 0x111111 }),
        );
        cap.position.set(0, 0, robotR * 0.82);
        robotMesh.add(cap);
        robotMesh.position.copy(pts[0]);
        robotGroup.add(robotMesh);
        const total = Math.max(1, pts.length - 1);
        robotsRef.current.push({
          points: pts,
          robot: robotMesh,
          frame: 0,
          maxLen: pts.length,
          speed: Math.max(0.25, total / 210),
          pauseEnd: total * 0.18,
          radius: robotR,
          up,
        });
      }
    }
    applyRobotRef.current = applyRobots;

    let rafId = 0;
    const tick = () => {
      rafId = requestAnimationFrame(tick);
      controls.update();
      for (const rs of robotsRef.current) {
        if (rs.maxLen <= 1) continue;
        const total = rs.maxLen - 1;
        rs.frame += rs.speed;
        if (rs.frame >= total + rs.pauseEnd) rs.frame = 0;
        const f = Math.min(rs.frame, total);
        const i0 = Math.floor(f);
        const i1 = Math.min(i0 + 1, total);
        const pos = rs.points[i0].clone().lerp(rs.points[i1], f - i0);
        const delta = pos.clone().sub(rs.robot.position);
        rs.robot.position.copy(pos);
        const d = delta.length();
        if (d > 1e-6) {
          const axis = new THREE.Vector3().crossVectors(rs.up, delta);
          if (axis.lengthSq() > 1e-12) {
            rs.robot.rotateOnWorldAxis(axis.normalize(), d / Math.max(rs.radius, 1e-4));
          }
        }
      }
      renderer.render(scene, camera);
    };
    tick();

    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      envRef.current = null;
      applyPlacementRef.current = null;
      applyRobotRef.current = null;
      robotsRef.current = [];
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Re-apply placement live when it changes (no mesh reload).
  useEffect(() => {
    applyPlacementRef.current?.();
  }, [placement]);

  // Rebuild the robots + paths when a new run arrives (no mesh reload).
  useEffect(() => {
    applyRobotRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robots.map((r) => r.key).join(",")]);

  return (
    <div
      className={cn(
        "relative w-full h-full border border-border rounded-sm overflow-hidden bg-[oklch(0.06_0_0)]",
        className,
      )}
    >
      <div ref={containerRef} className="absolute inset-0" />
      {load.state !== "ready" && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6 text-center">
          {load.state === "loading" ? (
            <span className="mono text-xs text-muted-foreground animate-pulse">
              Loading 3D scene…
            </span>
          ) : (
            <span className="mono text-xs text-[var(--status-fail)] max-w-xs">
              {load.msg ?? "Couldn't load the 3D scene."}
            </span>
          )}
        </div>
      )}
      <div className="absolute pointer-events-none top-3 right-3 mono text-[10px] text-muted-foreground/70 px-2 py-1 rounded-sm bg-background/60 backdrop-blur-sm">
        drag <span className="text-foreground">orbit</span> · shift+drag{" "}
        <span className="text-foreground">pan</span> · wheel{" "}
        <span className="text-foreground">zoom</span>
      </div>
    </div>
  );
}
