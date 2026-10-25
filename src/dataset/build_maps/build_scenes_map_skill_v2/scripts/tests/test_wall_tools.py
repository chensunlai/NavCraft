"""Small regression fixtures; no Habitat, GPU, network, or production writes."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_wall_layers import clip_walls, draw
from validate_wall_stage import validate
from refresh_manifest import refresh


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


class ClipTests(unittest.TestCase):
    def test_rotated_door_cuts_parallel_faces_preserves_return(self):
        wall = dict(id='W',level='Mezzanine',start=[0,0],end=[4,4],origin='observed_schematic')
        return_wall = dict(wall,id='Return',start=[2,2],end=[3,1])
        door = dict(id='D',level='Mezzanine',segment=[[1,1],[3,3]])
        source = dict(segments=[wall,return_wall],openings=[door])
        clipped = clip_walls(source,.18)
        self.assertEqual(len(clipped['segments']),3)
        self.assertEqual(clipped['segments'][-1]['end'],[3,1])
        self.assertEqual(source['segments'][0]['end'],[4,4])

    def test_floor_and_explicit_wall_ids_limit_clipping(self):
        source = dict(segments=[dict(id='A',level='B1',start=[0,0],end=[4,0]),dict(id='B',level='Roof',start=[0,0],end=[4,0])],
                      openings=[dict(id='D',level='B1',segment=[[1,0],[3,0]],wall_ids=['not-A'])])
        self.assertEqual([s['end'] for s in clip_walls(source,.18)['segments']],[[4,0],[4,0]])


class RenderTests(unittest.TestCase):
    def test_stair_labels_and_footprint_use_scene_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            scene=dict(scene_id='translated_fixture',levels=[
                dict(id='Basement',floor_height=-3,bounds=[[100,106],[100,105]]),
                dict(id='Mezzanine',floor_height=0,bounds=[[100,106],[100,105]])])
            stair=dict(id='Flight7',from_level='Basement',to_level='Mezzanine',
                       polygon=[[101,101],[102,101],[102,104],[101,104]],
                       treads=[[[101,102],[102,102]],[[101,103],[102,103]]],
                       ascent_points=[[101.5,101.2],[101.5,103.8]])
            draw(dict(segments=[],openings=[]),scene,[stair],output,'walls',60)
            text=(output/'walls.svg').read_text()
            self.assertIn('Flight7 UP TO Mezzanine',text)
            self.assertIn('Flight7 DOWN TO Basement',text)
            with Image.open(output/'walls.png') as image:
                image.verify()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'scene'
        self.raw = self.root/'raw'
        self.raw.mkdir(parents=True)
        (self.root/'evidence').mkdir()
        for path in [self.root/'floor_plan.png',self.root/'evidence_rooms.jpg',self.raw/'walls.png',self.raw/'wall_real.png',self.root/'evidence/F.png']:
            Image.new('RGB',(10,10),'white').save(path)
        (self.root/'scene_description.md').write_text('Fixture\n')
        self.scene = dict(scene_id='fixture',stage='walls_only',levels=[dict(id='Mezz',floor_height=2.5,bounds=[[-1,5],[-1,3]])],
                          files=dict(walls='walls.json',wall_real='wall_real.json',stairs='stairs.json',review='layout_review.json'))
        self.wall = dict(scene_id='fixture',unknown_areas=[],segments=[dict(id='W1',level='Mezz',start=[0,0],end=[1,0],status='resolved',origin='observed_schematic',kind='wall',basis='fixture observation',evidence_frames=['F']),
                                                                    dict(id='W2',level='Mezz',start=[3,0],end=[4,0],status='resolved',origin='observed_schematic',kind='wall',basis='fixture observation',evidence_frames=['F'])],
                         openings=[dict(id='D1',level='Mezz',segment=[[1,0],[3,0]],status='confirmed_open',evidence_frames=['F'])])
        for name,data in {'scene.json':self.scene,'walls.json':self.wall,'wall_real.json':self.wall,'doors.json':dict(doors=self.wall['openings']),
                          'stairs.json':dict(scene_id='fixture',stairs=[]),'evidence.json':dict(frames=[dict(id='F',source='offline_habitat_render',rgb='../evidence/F.png')]),
                          'layout_review.json':dict(scene_id='fixture'),'manifest.json':dict(scene_id='fixture',version=7,status='draft',checksum_path_base='..')}.items():
            save(self.raw/name,data)
        refresh(self.root,True)

    def kinds(self):
        return {e['kind'] for e in validate(self.root)['errors']}

    def test_valid_fixture(self):
        self.assertTrue(validate(self.root)['structural_pass'])
        self.assertIn('manual_review_required',{w['kind'] for w in validate(self.root)['warnings']})

    def test_estimate_cannot_enter_real(self):
        bad = copy.deepcopy(self.wall)
        bad['segments'][0]['origin']='inferred_completion'
        save(self.raw/'wall_real.json',bad)
        refresh(self.root,True)
        self.assertIn('estimate_in_wall_real',self.kinds())

    def test_pending_geometry_rejected(self):
        self.wall['segments'][0]['status']='estimated_occluded'
        save(self.raw/'walls.json',self.wall)
        refresh(self.root,True)
        self.assertIn('pending_wall_geometry',self.kinds())

    def test_blocked_portal_and_duplicate_detected(self):
        wall = copy.deepcopy(self.wall['segments'][0])
        wall.update(id='bad',end=[4,0])
        self.wall['segments'].append(wall)
        save(self.raw/'walls.json',self.wall)
        refresh(self.root,True)
        self.assertTrue({'portal_blocked','duplicate_wall_overlap'} <= self.kinds())

    def test_stale_report_survives_manifest_refresh(self):
        save(self.raw/'path_check.json',dict(inputs_sha256={'walls.json':'0'*64}))
        refresh(self.root,True)
        self.assertIn('stale_check',self.kinds())

    def test_manifest_preview_is_read_only_and_write_preserves_status(self):
        before=(self.raw/'manifest.json').read_bytes()
        (self.root/'scene_description.md').write_text('Changed\n')
        self.assertIn('checksum_mismatch',self.kinds())
        preview=refresh(self.root)
        self.assertEqual(before,(self.raw/'manifest.json').read_bytes())
        self.assertIn('scene_description.md',preview['changes']['changed'])
        refresh(self.root,True)
        manifest=json.loads((self.raw/'manifest.json').read_text())
        self.assertEqual((manifest['version'],manifest['status']),(7,'draft'))
        self.assertTrue(validate(self.root)['structural_pass'])

    def test_missing_file_reports_error(self):
        (self.root/'evidence/F.png').unlink()
        self.assertIn('missing_reference',self.kinds())

    def test_reference_escape_is_rejected(self):
        save(self.raw/'evidence.json',dict(frames=[dict(id='F',source='x',rgb='../../outside.png')]))
        with self.assertRaisesRegex(ValueError,'escapes archive'):
            validate(self.root)


if __name__ == '__main__':
    unittest.main()
